// Copyright (c) 2014-2015 Tero Hänninen, license: MIT

extern crate flate2;
use std::io::{self, Read, Write};
use std::iter::{repeat};
use std::cmp::min;
use self::flate2::read::{ZlibDecoder, ZlibEncoder};
use self::flate2::Compression;
use super::{
    Image, Info, ColFmt, ColType, error,
    copy_memory, get_converter,
    u32_to_be, u32_from_be, Crc32, crc32be, IFRead,
};

/// Header of a PNG image.
#[derive(Debug)]
pub struct PngHeader {
    pub width              : u32,
    pub height             : u32,
    pub bit_depth          : u8,
    pub color_type         : u8,
    pub compression_method : u8,
    pub filter_method      : u8,
    pub interlace_method   : u8
}

static PNG_FILE_HEADER: [u8; 8] =
    [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

/// Returns width, height and color type of the image.
pub fn read_png_info<R: Read>(reader: &mut R) -> io::Result<Info> {
    let hdr = try!(read_png_header(reader));

    Ok(Info {
        w: hdr.width as usize,
        h: hdr.height as usize,
        ct: match hdr.color_type {
               0 => ColType::Gray,
               2 => ColType::Color,
               3 => ColType::Color,   // type of the palette
               4 => ColType::GrayAlpha,
               6 => ColType::ColorAlpha,
               _ => ColType::Auto,      // unknown type
           },
    })
}

/// Reads a PNG header.
///
/// The fields are not parsed into enums or anything like that.
pub fn read_png_header<R: Read>(reader: &mut R) -> io::Result<PngHeader> {
    let mut buf = [0u8; 33];  // file header + IHDR
    try!(reader.read_exact(&mut buf));

    if &buf[0..8] != &PNG_FILE_HEADER[..] ||
       &buf[8..16] != b"\0\0\0\x0dIHDR" ||
       &buf[29..33] != &crc32be(&buf[12..29])[..]
    {
        return error("corrupt png header");
    }

    Ok(PngHeader {
        width              : u32_from_be(&buf[16..20]),
        height             : u32_from_be(&buf[20..24]),
        bit_depth          : buf[24],
        color_type         : buf[25],
        compression_method : buf[26],
        filter_method      : buf[27],
        interlace_method   : buf[28],
    })
}

/// Reads an image and converts it to requested format.
///
/// Passing `ColFmt::Auto` as `req_fmt` converts the data to one of `Y`, `YA`, `RGB`,
/// `RGBA`.  Paletted images are auto-depaletted.
#[inline]
pub fn read_png<R: Read>(reader: &mut R, req_fmt: ColFmt) -> io::Result<Image> {
    let (image, _) = try!(read_png_chunks(reader, req_fmt, &[]));
    Ok(image)
}

/// Like `read_png` but also returns the requested extension chunks.
///
/// If the requested chunks are not present they are ignored.
pub fn read_png_chunks<R: Read>(reader: &mut R, req_fmt: ColFmt, chunk_names: &[[u8; 4]])
                       -> io::Result<(Image, Vec<PngCustomChunk>)>
{
    let hdr = try!(read_png_header(reader));

    if hdr.width < 1 || hdr.height < 1 { return error("invalid dimensions") }
    if hdr.bit_depth != 8 { return error("only 8-bit images supported") }
    if hdr.compression_method != 0 || hdr.filter_method != 0 {
        return error("not supported");
    }

    let ilace = match PngInterlace::from_u8(hdr.interlace_method) {
        Some(im) => im,
        None => return error("unsupported interlace method"),
    };

    let src_fmt = match hdr.color_type {
        0 => ColFmt::Y,
        2 => ColFmt::RGB,
        3 => ColFmt::RGB,   // format of the palette
        4 => ColFmt::YA,
        6 => ColFmt::RGBA,
        _ => return error("unsupported color type"),
    };

    let dc = &mut PngDecoder {
        stream      : reader,
        w           : hdr.width as usize,
        h           : hdr.height as usize,
        ilace       : ilace,
        src_indexed : hdr.color_type == PngColortype::Idx as u8,
        src_fmt     : src_fmt,
        tgt_fmt     : if req_fmt == ColFmt::Auto { src_fmt } else { req_fmt },
        chunk_lentype : [0u8; 8],
        readbuf     : repeat(0u8).take(4096).collect(),
        crc         : Crc32::new(),
    };

    let (buf, chunks) = try!(decode_png(dc, chunk_names));
    Ok((Image {
        w   : dc.w,
        h   : dc.h,
        fmt : dc.tgt_fmt,
        buf : buf
    }, chunks))
}

struct PngDecoder<'r, R:'r> {
    stream        : &'r mut R,
    w             : usize,
    h             : usize,
    ilace         : PngInterlace,
    src_indexed   : bool,
    src_fmt       : ColFmt,
    tgt_fmt       : ColFmt,

    chunk_lentype: [u8; 8],   // for reading len, type
    readbuf: Vec<u8>,
    crc: Crc32,
}

#[derive(Eq, PartialEq)]
enum PngStage {
    IhdrParsed,
    PlteParsed,
    IdatParsed,
    //IendParsed,
}

fn read_chunkmeta<R: Read>(dc: &mut PngDecoder<R>) -> io::Result<usize> {
    try!(dc.stream.read_exact(&mut dc.chunk_lentype[0..8]));
    let len = u32_from_be(&dc.chunk_lentype[0..4]) as usize;
    if 0x7fff_ffff < len { return error("chunk too long"); }
    dc.crc.put(&dc.chunk_lentype[4..8]);   // type
    Ok(len)
}

#[inline]
fn readcheck_crc<R: Read>(dc: &mut PngDecoder<R>) -> io::Result<()> {
    let mut tmp = [0u8; 4];
    try!(dc.stream.read_exact(&mut tmp));
    if &dc.crc.finish_be()[..] != &tmp[0..4] {
        return error("corrupt chunk");
    }
    Ok(())
}

fn decode_png<R: Read>(dc: &mut PngDecoder<R>, chunk_names: &[[u8; 4]])
                              -> io::Result<(Vec<u8>, Vec<PngCustomChunk>)>
{
    use self::PngStage::*;

    let mut result = Vec::<u8>::new();
    let mut chunks = Vec::<PngCustomChunk>::new();
    let mut palette = Vec::<u8>::new();

    let mut stage = IhdrParsed;

    let mut len = try!(read_chunkmeta(dc));

    loop {
        match &dc.chunk_lentype[4..8] {
            b"IDAT" => {
                if !(stage == IhdrParsed || (stage == PlteParsed && dc.src_indexed)) {
                    return error("corrupt chunk stream");
                }

                // also reads chunk_lentype for next chunk
                result = try!(read_idat_stream(dc, &mut len, &palette[..]));
                stage = IdatParsed;
                continue;   // skip reading chunk_lentype
            }
            b"PLTE" => {
                let entries = len / 3;
                if stage != IhdrParsed || len % 3 != 0 || 256 < entries {
                    return error("corrupt chunk stream");
                }
                palette = repeat(0u8).take(len).collect();
                try!(dc.stream.read_exact(&mut palette));
                dc.crc.put(&palette[..]);
                try!(readcheck_crc(dc));
                stage = PlteParsed;
            }
            b"IEND" => {
                if stage != IdatParsed {
                    return error("corrupt chunk stream");
                }
                let mut crc = [0u8; 4];
                try!(dc.stream.read_exact(&mut crc));
                if len != 0 || &crc[0..4] != &[0xae, 0x42, 0x60, 0x82][..] {
                    return error("corrupt chunk");
                }
                break;//stage = IendParsed;
            }
            _ => {
                if chunk_names.iter().any(|name| &name[..] == &dc.chunk_lentype[4..8]) {
                    let name = [dc.chunk_lentype[4], dc.chunk_lentype[5],
                                dc.chunk_lentype[6], dc.chunk_lentype[7]];
                    let mut data: Vec<u8> = repeat(0u8).take(len).collect();
                    try!(dc.stream.read_exact(&mut data));
                    dc.crc.put(&data[..]);
                    chunks.push(PngCustomChunk { name: name, data: data });
                } else {
                    // unknown chunk, ignore but check crc... or should crc be ignored?
                    while 0 < len {
                        let amount = min(len, dc.readbuf.len());
                        try!(dc.stream.read_exact(&mut dc.readbuf[0..amount]));
                        len -= amount;
                        dc.crc.put(&dc.readbuf[0..amount]);
                    }
                }

                try!(readcheck_crc(dc));
            }
        }

        len = try!(read_chunkmeta(dc));
    }

    Ok((result, chunks))
}

#[derive(Eq, PartialEq)]
enum PngInterlace {
    None, Adam7
}

impl PngInterlace {
    fn from_u8(val: u8) -> Option<PngInterlace> {
        match val {
            0 => Some(PngInterlace::None),
            1 => Some(PngInterlace::Adam7),
            _ => None,
        }
    }
}

#[derive(Eq, PartialEq)]
enum PngColortype {
    Y    = 0,
    RGB  = 2,
    Idx  = 3,
    YA   = 4,
    RGBA = 6,
}

fn depalettize(src: &[u8], palette: &[u8], dst: &mut[u8]) {
    let mut d = 0;
    for &pi in src {
        let pidx = pi as usize * 3;
        copy_memory(&palette[pidx..pidx+3], &mut dst[d..d+3]);
        d += 3;
    }
}

fn read_idat_stream<R: Read>(dc: &mut PngDecoder<R>, len: &mut usize, palette: &[u8])
                                                               -> io::Result<Vec<u8>>
{
    let filter_step = if dc.src_indexed { 1 } else { dc.src_fmt.bytes_pp() };
    let tgt_bytespp = dc.tgt_fmt.bytes_pp();
    let tgt_linesize = dc.w * tgt_bytespp;

    let mut result: Vec<u8> = repeat(0).take(dc.w * dc.h * tgt_bytespp).collect();
    let mut depaletted_line: Vec<u8> = if dc.src_indexed {
        repeat(0).take(dc.w * 3).collect()
    } else {
        Vec::new()
    };

    let convert = match get_converter(dc.src_fmt, dc.tgt_fmt) {
        Some(c) => c,
        None => return error("no such converter"),
    };

    let compressed_data = try!(read_idat_chunks(dc, len));
    let mut zlib = ZlibDecoder::new(&compressed_data[..]);

    match dc.ilace {
        PngInterlace::None => {
            let src_linesize = dc.w * filter_step;
            let mut cline: Vec<u8> = repeat(0).take(src_linesize+1).collect();
            let mut pline: Vec<u8> = repeat(0).take(src_linesize+1).collect();

            let mut ti = 0;
            for _j in (0 .. dc.h) {
                try!(zlib.read_exact(&mut cline[..]));
                let filter_type: u8 = cline[0];

                try!(recon(
                    &mut cline[1 .. src_linesize+1], &pline[1 .. src_linesize+1],
                    filter_type, filter_step
                ));

                if dc.src_indexed {
                    depalettize(&cline[1..], &palette, &mut depaletted_line);
                    convert(&depaletted_line[0 .. src_linesize*3],
                            &mut result[ti .. ti+tgt_linesize])
                } else {
                    convert(&cline[1..], &mut result[ti .. ti+tgt_linesize]);
                }

                ti += tgt_linesize;

                let swap = pline;
                pline = cline;
                cline = swap;
            }
        },
        PngInterlace::Adam7 => {
            let redw: [usize; 7] = [
                (dc.w + 7) / 8,
                (dc.w + 3) / 8,
                (dc.w + 3) / 4,
                (dc.w + 1) / 4,
                (dc.w + 1) / 2,
                (dc.w + 0) / 2,
                (dc.w + 0) / 1,
            ];
            let redh: [usize; 7] = [
                (dc.h + 7) / 8,
                (dc.h + 7) / 8,
                (dc.h + 3) / 8,
                (dc.h + 3) / 4,
                (dc.h + 1) / 4,
                (dc.h + 1) / 2,
                (dc.h + 0) / 2,
            ];

            let max_scanline_size = dc.w * filter_step;
            let mut linebuf0: Vec<u8> = repeat(0).take(max_scanline_size+1).collect();
            let mut linebuf1: Vec<u8> = repeat(0).take(max_scanline_size+1).collect();
            let mut redlinebuf: Vec<u8> = repeat(0).take(dc.w * tgt_bytespp).collect();

            for pass in (0..7) {
                let tgt_px: A7IdxTranslator = A7_IDX_TRANSLATORS[pass];   // target pixel
                let src_linesize = redw[pass] * filter_step;

                for j in (0 .. redh[pass]) {
                    let (cline, pline) = if j % 2 == 0 {
                        (&mut linebuf0[0 .. src_linesize+1],
                        &mut linebuf1[0 .. src_linesize+1])
                    } else {
                        (&mut linebuf1[0 .. src_linesize+1],
                        &mut linebuf0[0 .. src_linesize+1])
                    };

                    try!(zlib.read_exact(&mut cline[..]));
                    let filter_type: u8 = cline[0];

                    try!(recon(&mut cline[1..], &pline[1..], filter_type, filter_step));

                    if dc.src_indexed {
                        depalettize(&cline[1..], &palette, &mut depaletted_line);
                        convert(&depaletted_line[0 .. src_linesize*3],
                                &mut redlinebuf[0..redw[pass] * tgt_bytespp])
                    } else {
                        convert(&cline[1..], &mut redlinebuf[0..redw[pass] * tgt_bytespp]);
                    }

                    let mut redi = 0;
                    for i in (0 .. redw[pass]) {
                        let tgt = tgt_px(i, j, dc.w) * tgt_bytespp;
                        copy_memory(
                            &redlinebuf[redi .. redi+tgt_bytespp],
                            &mut result[tgt .. tgt+tgt_bytespp]
                        );
                        redi += tgt_bytespp;
                    }

                }
            }
        } // Adam7
    }

    return Ok(result);
}

type A7IdxTranslator = fn(redx: usize, redy: usize, dstw: usize) -> usize;
static A7_IDX_TRANSLATORS: [A7IdxTranslator; 7] = [
    a7_red1_to_dst,
    a7_red2_to_dst,
    a7_red3_to_dst,
    a7_red4_to_dst,
    a7_red5_to_dst,
    a7_red6_to_dst,
    a7_red7_to_dst,
];

fn a7_red1_to_dst(redx:usize, redy:usize, dstw:usize) -> usize { redy*8*dstw + redx*8     }
fn a7_red2_to_dst(redx:usize, redy:usize, dstw:usize) -> usize { redy*8*dstw + redx*8+4   }
fn a7_red3_to_dst(redx:usize, redy:usize, dstw:usize) -> usize { (redy*8+4)*dstw + redx*4 }
fn a7_red4_to_dst(redx:usize, redy:usize, dstw:usize) -> usize { redy*4*dstw + redx*4+2   }
fn a7_red5_to_dst(redx:usize, redy:usize, dstw:usize) -> usize { (redy*4+2)*dstw + redx*2 }
fn a7_red6_to_dst(redx:usize, redy:usize, dstw:usize) -> usize { redy*2*dstw + redx*2+1   }
fn a7_red7_to_dst(redx:usize, redy:usize, dstw:usize) -> usize { (redy*2+1)*dstw + redx   }

// will leave len to the length of next chunk after last idat chunk
fn read_idat_chunks<R: Read>(dc: &mut PngDecoder<R>, len: &mut usize) -> io::Result<(Vec<u8>)> {
    let mut all: Vec<u8> = Vec::new();
    loop {
        all.extend(repeat(0).take(*len));
        let start = all.len() - *len;
        try!(dc.stream.read_exact(&mut all[start..]));
        dc.crc.put(&all[start..]);
        try!(readcheck_crc(dc));
        *len = try!(read_chunkmeta(dc));    // next chunk's len and type
        if &dc.chunk_lentype[4..8] != b"IDAT" {
            break;
        }
    }
    Ok(all)
}

fn recon(cline: &mut[u8], pline: &[u8], ftype: u8, fstep: usize) -> io::Result<()> {
    match PngFilter::from_u8(ftype) {
        Some(PngFilter::None)
            => { }
        Some(PngFilter::Sub) => {
            unsafe {
                for k in (fstep .. cline.len()) {
                    *cline.get_unchecked_mut(k) =
                        (*cline.get_unchecked(k))
                        .wrapping_add(*cline.get_unchecked(k-fstep));
                }
            }
        }
        Some(PngFilter::Up) => {
            for (c, &p) in cline.iter_mut().zip(pline) {
                *c = c.wrapping_add(p);
            }
        }
        Some(PngFilter::Average) => {
            for (c, &p) in (&mut cline[..fstep]).iter_mut()
                            .zip(&pline[..fstep])
            {
                *c = c.wrapping_add(p / 2);
            }
            unsafe {
                for k in (fstep .. cline.len()) {
                    *cline.get_unchecked_mut(k) = (*cline.get_unchecked(k))
                        .wrapping_add(((*cline.get_unchecked(k-fstep) as u32
                                    + *pline.get_unchecked(k) as u32) / 2) as u8);
                }
            }
        }
        Some(PngFilter::Paeth) => {
            for (c, &p) in (&mut cline[..fstep]).iter_mut().zip(&pline[..fstep]) {
                *c = c.wrapping_add(paeth(0, p, 0));
            }
            unsafe {
                for k in (fstep .. cline.len()) {
                    *cline.get_unchecked_mut(k) =
                        (*cline.get_unchecked(k)).wrapping_add(
                            paeth(*cline.get_unchecked(k-fstep),
                                        *pline.get_unchecked(k),
                                          *pline.get_unchecked(k-fstep)));
                }
            }
        }
        None => return error("filter type not supported"),
    }
    Ok(())
}

fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let mut pc = c as i32;
    let mut pa = b as i32 - pc;
    let mut pb = a as i32 - pc;
    pc = pa + pb;
    if pa < 0 { pa = -pa; }
    if pb < 0 { pb = -pb; }
    if pc < 0 { pc = -pc; }

    if pa <= pb && pa <= pc {
        return a;
    } else if pb <= pc {
        return b;
    }
    return c;
}

enum PngFilter {
    None = 0,
    Sub,
    Up,
    Average,
    Paeth,
}

impl PngFilter {
    fn from_u8(val: u8) -> Option<PngFilter> {
        match val {
            0 => Some(PngFilter::None),
            1 => Some(PngFilter::Sub),
            2 => Some(PngFilter::Up),
            3 => Some(PngFilter::Average),
            4 => Some(PngFilter::Paeth),
            _ => None,
        }
    }
}

// --------------------------------------------------
// PNG encoder

/// PNG extension chunk.
pub struct PngCustomChunk {
    pub name: [u8; 4],
    pub data: Vec<u8>,
}

/// Writes an image and converts it to requested color type.
#[inline]
pub fn write_png<W: Write>(writer: &mut W, w: usize, h: usize, src_fmt: ColFmt,
                                                                   data: &[u8],
                                                             tgt_type: ColType)
                                                              -> io::Result<()>
{
    write_png_chunks(writer, w, h, src_fmt, data, tgt_type, &[])
}

/// Like `write_png` but also writes the given extension chunks.
pub fn write_png_chunks<W: Write>(writer: &mut W, w: usize, h: usize, src_fmt: ColFmt,
                                                                          data: &[u8],
                                                                    tgt_type: ColType,
                                                            chunks: &[PngCustomChunk])
                                                                     -> io::Result<()>
{
    let src_bytespp = data.len() / w / h;

    if w < 1 || h < 1
    || src_bytespp * w * h != data.len()
    || src_bytespp != src_fmt.bytes_pp() {
        return error("invalid dimensions or data length");
    }

    match src_fmt {
        ColFmt::Y | ColFmt::YA | ColFmt::RGB | ColFmt::RGBA
                                             | ColFmt::BGR
                                             | ColFmt::BGRA => {}
        ColFmt::Auto => return error("invalid format"),
    }

    let tgt_fmt = match tgt_type {
        ColType::Gray       => ColFmt::Y,
        ColType::GrayAlpha  => ColFmt::YA,
        ColType::Color      => ColFmt::RGB,
        ColType::ColorAlpha => ColFmt::RGBA,
        ColType::Auto => match src_fmt {
            ColFmt::Y    | ColFmt::YA => src_fmt,
            ColFmt::RGB  | ColFmt::BGR => ColFmt::RGB,
            ColFmt::RGBA | ColFmt::BGRA => ColFmt::RGBA,
            ColFmt::Auto => return error("invalid format"),
        },
    };

    let ec = &mut PngEncoder {
        stream    : writer,
        w         : w,
        h         : h,
        src_bytespp : src_bytespp,
        src_fmt   : src_fmt,
        tgt_fmt   : tgt_fmt,
        data      : data,
        crc       : Crc32::new(),
    };

    try!(write_png_header(ec));
    for chunk in chunks {
        try!(write_custom_chunk(ec, chunk));
    }
    try!(write_png_image_data(ec));

    let iend: &'static[u8] = b"\0\0\0\0IEND\xae\x42\x60\x82";
    ec.stream.write_all(iend)
}

fn write_png_header<W: Write>(ec: &mut PngEncoder<W>) -> io::Result<()> {
    let mut hdr: [u8; 33] = [0; 33];

    copy_memory(&PNG_FILE_HEADER[..]       , &mut hdr[0..8]  );
    copy_memory(b"\0\0\0\x0dIHDR"          , &mut hdr[8..16] );
    copy_memory(&u32_to_be(ec.w as u32)[..], &mut hdr[16..20]);
    copy_memory(&u32_to_be(ec.h as u32)[..], &mut hdr[20..24]);
    hdr[24] = 8;    // bit depth
    hdr[25] = match ec.tgt_fmt {    // color type
        ColFmt::Y => PngColortype::Y,
        ColFmt::YA => PngColortype::YA,
        ColFmt::RGB => PngColortype::RGB,
        ColFmt::RGBA => PngColortype::RGBA,
        _ => return error("not supported"),
    } as u8;
    copy_memory(&[0, 0, 0], &mut hdr[26..29]);  // compression, filter, interlace
    ec.crc.put(&hdr[12..29]);
    copy_memory(&ec.crc.finish_be()[..], &mut hdr[29..33]);

    ec.stream.write_all(&hdr[..])
}

fn write_custom_chunk<W: Write>(ec: &mut PngEncoder<W>, chunk: &PngCustomChunk) -> io::Result<()> {
    if chunk.name[0] < 97 || 122 < chunk.name[0] { return error("invalid chunk name"); }
    for b in &chunk.name[1..] {
        if *b < 65 || (90 < *b && *b < 97) || 122 < *b {
            return error("invalid chunk name");
        }
    }
    if 0x7fff_ffff < chunk.data.len() { return error("chunk too long"); }

    try!(ec.stream.write_all(&u32_to_be(chunk.data.len() as u32)[..]));
    try!(ec.stream.write_all(&chunk.name[..]));
    try!(ec.stream.write_all(&chunk.data[..]));
    let mut crc = Crc32::new();
    crc.put(&chunk.name[..]);
    crc.put(&chunk.data[..]);
    ec.stream.write_all(&crc.finish_be()[..])
}

struct PngEncoder<'r, W:'r> {
    stream        : &'r mut W,
    w             : usize,
    h             : usize,
    src_bytespp   : usize,
    tgt_fmt       : ColFmt,
    src_fmt       : ColFmt,
    data          : &'r [u8],
    crc           : Crc32,
}

fn write_png_image_data<W: Write>(ec: &mut PngEncoder<W>) -> io::Result<()> {
    let convert = match get_converter(ec.src_fmt, ec.tgt_fmt) {
        Some(c) => c,
        None => return error("no such converter"),
    };

    let filter_step = ec.tgt_fmt.bytes_pp();
    let tgt_linesize = ec.w * filter_step + 1;   // +1 for filter type
    let mut cline: Vec<u8> = repeat(0).take(tgt_linesize).collect();
    let mut pline: Vec<u8> = repeat(0).take(tgt_linesize).collect();
    let mut filtered_image: Vec<u8> = repeat(0).take(tgt_linesize * ec.h).collect();

    let src_linesize = ec.w * ec.src_bytespp;

    let mut si = 0;
    let mut ti = 0;
    while si < ec.h * src_linesize {
        convert(&ec.data[si .. si+src_linesize], &mut cline[1 .. tgt_linesize]);

        for i in (1 .. filter_step+1) {
            filtered_image[ti+i] = cline[i].wrapping_sub(paeth(0, pline[i], 0));
        }
        for i in (filter_step+1 .. cline.len()) {
            filtered_image[ti+i] = cline[i]
                .wrapping_sub(paeth(cline[i-filter_step], pline[i], pline[i-filter_step]));
        }

        filtered_image[ti] = PngFilter::Paeth as u8;

        let swap = pline;
        pline = cline;
        cline = swap;

        si += src_linesize;
        ti += tgt_linesize;
    }

    // TODO review...
    let mut zlibenc = ZlibEncoder::new(&filtered_image[..], Compression::Fast);
    let mut compressed = Vec::<u8>::new();
    try!(zlibenc.read_to_end(&mut compressed));
    ec.crc.put(b"IDAT");
    ec.crc.put(&compressed[..]);
    let crc = &ec.crc.finish_be();

    // TODO split up data into smaller chunks
    try!(ec.stream.write_all(&u32_to_be(compressed.len() as u32)[..]));
    try!(ec.stream.write_all(b"IDAT"));
    try!(ec.stream.write_all(&compressed[..]));
    try!(ec.stream.write_all(crc));
    Ok(())
}

