/*
    Copyright (c) 2014 Tero Hänninen

    MIT Software License

    Permission is hereby granted, free of charge, to any person obtaining a copy of
    this software and associated documentation files (the "Software"), to deal in
    the Software without restriction, including without limitation the rights to
    use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
    the Software, and to permit persons to whom the Software is furnished to do so,
    subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
    FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
    COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
    IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
    CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

extern crate flate;
use std::old_io::{File, BufferedReader, BufferedWriter, IoResult, IoError, OtherIoError};
use std::iter::{repeat, range_step, IteratorExt};
use std::cmp::min;
use std::slice::bytes::{copy_memory};
use std::mem::{zeroed};
use std::num::{Float, FromPrimitive};
use self::flate::{inflate_bytes_zlib, deflate_bytes_zlib};

macro_rules! IFErr {
    ($e:expr) => (Err(IoError{kind: OtherIoError, desc: $e, detail: None}))
}

#[derive(Debug)]
pub struct IFImage {
    pub w      : usize,
    pub h      : usize,
    pub c      : ColFmt,
    pub pixels : Vec<u8>,
}

#[derive(Debug)]
pub struct IFInfo {
    pub w : usize,
    pub h : usize,
    pub c : ColFmt,
}

#[derive(Copy, Debug, Eq, PartialEq)]
pub enum ColFmt {
    Auto = 0,
    Y = 1,
    YA,
    RGB,
    RGBA,
    BGR,
    BGRA,
}

/** Returns: basic info about an image file. The color format information does
 * not correspond to the exact format in the file: for BGR/A data the format is
 * reported as RGB/A and for paletted images it might be RGB or RGBA or
 * whatever (paletted images are auto-depaletted by the decoders).  */
#[allow(dead_code)]
pub fn read_image_info(filename: &str) -> IoResult<IFInfo> {
    type F = fn(&mut BufferedReader<File>) -> IoResult<IFInfo>;
    let readfunc: F = match extract_extension(filename) {
        Some(".png")                 => read_png_info,
        Some(".tga")                 => read_tga_info,
        Some(".jpg") | Some(".jpeg") => read_jpeg_info,
        _ => return IFErr!("extension not recognized"),
    };
    let file = try!(File::open(&Path::new(filename)));
    let reader = &mut BufferedReader::new(file);
    readfunc(reader)
}

/** Paletted images are auto-depaletted.  */
#[allow(dead_code)]
pub fn read_image(filename: &str, req_fmt: ColFmt) -> IoResult<IFImage> {
    type F = fn(&mut BufferedReader<File>, ColFmt) -> IoResult<IFImage>;
    let readfunc: F = match extract_extension(filename) {
        Some(".png")                 => read_png,
        Some(".tga")                 => read_tga,
        Some(".jpg") | Some(".jpeg") => read_jpeg,
        _ => return IFErr!("extension not recognized"),
    };
    let file = try!(File::open(&Path::new(filename)));
    let reader = &mut BufferedReader::new(file);
    readfunc(reader, req_fmt)
}

#[allow(dead_code)]
pub fn write_image(filename: &str, w: usize, h: usize, data: &[u8], tgt_fmt: ColFmt)
                                                                     -> IoResult<()>
{
    type F = fn(&mut BufferedWriter<File>, usize, usize, &[u8], ColFmt) -> IoResult<()>;
    let writefunc: F = match extract_extension(filename) {
        Some(".png") => write_png,
        Some(".tga") => write_tga,
        _ => return IFErr!("extension not supported for writing"),
    };
    let file = try!(File::create(&Path::new(filename)));
    let writer = &mut BufferedWriter::new(file);
    writefunc(writer, w, h, data, tgt_fmt)
}

impl ColFmt {
    fn channels(&self) -> usize {
        use self::ColFmt::*;
        match *self {
            Auto        => 0,
            Y           => 1,
            YA          => 2,
            RGB  | BGR  => 3,
            RGBA | BGRA => 4,
        }
    }
}

// ------------------------------------------------------------

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

pub fn read_png_info<R: Reader>(reader: &mut R) -> IoResult<IFInfo> {
    let hdr = try!(read_png_header(reader));

    let ctype: Option<PngColortype> = FromPrimitive::from_u8(hdr.color_type);
    let ctype = match ctype {
        Some(ct) => ct,
        None => return IFErr!("unsupported color type"),
    };

    let src_fmt = match ctype {
        PngColortype::Y => ColFmt::Y,
        PngColortype::YA => ColFmt::YA,
        PngColortype::RGB => ColFmt::RGB,
        PngColortype::Idx => ColFmt::RGB,
        PngColortype::RGBA => ColFmt::RGBA,
    };

    Ok(IFInfo {
        w: hdr.width as usize,
        h: hdr.height as usize,
        c: src_fmt,
    })
}

pub fn read_png_header<R: Reader>(reader: &mut R) -> IoResult<PngHeader> {
    let mut buf = [0u8; 33];  // file header + IHDR
    try!(reader.read_at_least(buf.len(), &mut buf));

    if &buf[0..8] != &PNG_FILE_HEADER[] ||
       &buf[8..16] != b"\0\0\0\x0dIHDR" ||
       &buf[29..33] != &crc32be(&buf[12..29])[]
    {
        return IFErr!("corrupt png header");
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

pub fn read_png<R: Reader>(reader: &mut R, req_fmt: ColFmt) -> IoResult<IFImage> {
    let (image, _) = try!(read_png_chunks(reader, req_fmt, &[]));
    Ok(image)
}

pub fn read_png_chunks<R: Reader>(reader: &mut R, req_fmt: ColFmt, chunk_names: &[[u8; 4]])
                       -> IoResult<(IFImage, Vec<PngCustomChunk>)>
{
    let req_fmt = { use self::ColFmt::*; match req_fmt {
        Auto | Y | YA | RGB | RGBA => req_fmt,
        _ => return IFErr!("format not supported")
    }};

    let hdr = try!(read_png_header(reader));

    if hdr.width < 1 || hdr.height < 1 { return IFErr!("invalid dimensions") }
    if hdr.bit_depth != 8 { return IFErr!("only 8-bit images supported") }
    if hdr.compression_method != 0 || hdr.filter_method != 0 {
        return IFErr!("not supported");
    }

    let ilace: PngInterlace = match FromPrimitive::from_u8(hdr.interlace_method) {
        Some(im) => im,
        None => return IFErr!("unsupported interlace method"),
    };

    let ctype: PngColortype = match FromPrimitive::from_u8(hdr.color_type) {
        Some(ct) => ct,
        None => return IFErr!("unsupported color type"),
    };

    let src_fmt = match ctype {
        PngColortype::Y => ColFmt::Y,
        PngColortype::YA => ColFmt::YA,
        PngColortype::RGB => ColFmt::RGB,
        PngColortype::Idx => ColFmt::RGB,
        PngColortype::RGBA => ColFmt::RGBA,
    };

    let dc = &mut PngDecoder {
        stream      : reader,
        w           : hdr.width as usize,
        h           : hdr.height as usize,
        ilace       : ilace,
        src_indexed : ctype == PngColortype::Idx,
        src_fmt     : src_fmt,
        tgt_fmt     : if req_fmt == ColFmt::Auto { src_fmt } else { req_fmt },
        chunk_lentype : [0u8; 8],
        chunk_crc   : [0u8; 4],
        readbuf     : repeat(0u8).take(4096).collect(),
        uc_buf      : Vec::<u8>::new(),
        uc_start    : 0,
        crc         : Crc32::new(),
    };

    let (pixels, chunks) = try!(decode_png(dc, chunk_names));
    Ok((IFImage {
        w      : dc.w,
        h      : dc.h,
        c      : dc.tgt_fmt,
        pixels : pixels
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
    chunk_crc: [u8; 4],     // for reading crc
    readbuf: Vec<u8>,
    uc_buf: Vec<u8>,
    uc_start: usize,
    crc: Crc32,
}

#[derive(Eq, PartialEq)]
enum PngStage {
    IhdrParsed,
    PlteParsed,
    IdatParsed,
    //IendParsed,
}

fn read_chunkmeta<R: Reader>(dc: &mut PngDecoder<R>) -> IoResult<usize> {
    try!(dc.stream.read_at_least(8, &mut dc.chunk_lentype[0..8]));
    let len = u32_from_be(&dc.chunk_lentype[0..4]) as usize;
    if 0x7fff_ffff < len { return IFErr!("chunk too long"); }
    dc.crc.put(&dc.chunk_lentype[4..8]);   // type
    Ok(len)
}

fn decode_png<R: Reader>(dc: &mut PngDecoder<R>, chunk_names: &[[u8; 4]])
                              -> IoResult<(Vec<u8>, Vec<PngCustomChunk>)>
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
                    return IFErr!("corrupt chunk stream");
                }

                // also reads chunk_lentype for next chunk
                result = try!(read_idat_stream(dc, &mut len, &palette[]));
                stage = IdatParsed;
                continue;   // skip reading chunk_lentype
            }
            b"PLTE" => {
                let entries = len / 3;
                if stage != IhdrParsed || len % 3 != 0 || 256 < entries {
                    return IFErr!("corrupt chunk stream");
                }
                palette = try!(dc.stream.read_exact(len));
                dc.crc.put(&palette[]);
                try!(dc.stream.read_at_least(4, &mut dc.chunk_crc[0..4]));
                if &dc.crc.finish_be()[] != &dc.chunk_crc[0..4] {
                    return IFErr!("corrupt chunk");
                }
                stage = PlteParsed;
            }
            b"IEND" => {
                if stage != IdatParsed {
                    return IFErr!("corrupt chunk stream");
                }
                try!(dc.stream.read_at_least(4, &mut dc.chunk_crc[0..4]));
                if len != 0 || &dc.chunk_crc[0..4] != &[0xae, 0x42, 0x60, 0x82][] {
                    return IFErr!("corrupt chunk");
                }
                break;//stage = IendParsed;
            }
            _ => {
                if chunk_names.iter().any(|name| &name[] == &dc.chunk_lentype[4..8]) {
                    let name = [dc.chunk_lentype[4], dc.chunk_lentype[5],
                                dc.chunk_lentype[6], dc.chunk_lentype[7]];
                    let data = try!(dc.stream.read_exact(len));
                    dc.crc.put(&data[]);
                    chunks.push(PngCustomChunk { name: name, data: data });
                } else {
                    // unknown chunk, ignore but check crc... or should crc be ignored?
                    while 0 < len {
                        let bytes = min(len, dc.readbuf.len());
                        let got =
                            try!(dc.stream.read_at_least(bytes, &mut dc.readbuf[0..bytes]));
                        len -= got;
                        dc.crc.put(&dc.readbuf[0..got]);
                    }
                }

                try!(dc.stream.read_at_least(4, &mut dc.chunk_crc[0..4]));
                if &dc.crc.finish_be()[] != &dc.chunk_crc[0..4] {
                    return IFErr!("corrupt chunk");
                }
            }
        }

        len = try!(read_chunkmeta(dc));
    }

    Ok((result, chunks))
}

#[derive(Eq, PartialEq, FromPrimitive)]
enum PngInterlace {
    None, Adam7
}

#[derive(Eq, PartialEq, FromPrimitive)]
enum PngColortype {
    Y    = 0,
    RGB  = 2,
    Idx  = 3,
    YA   = 4,
    RGBA = 6,
}

fn depalette_convert(src_line: &[u8], tgt_line: &mut[u8], palette: &[u8],
                                               depaletted_line: &mut[u8],
                                             chan_convert: LineConverter)
{
    let mut d = 0us;
    for s in (0 .. src_line.len()) {
        let pidx = src_line[s] as usize * 3;
        copy_memory(&mut depaletted_line[d..d+3], &palette[pidx..pidx+3]);
        d += 3;
    }
    chan_convert(&depaletted_line[0 .. src_line.len()*3], tgt_line)
}

fn simple_convert(src_line: &[u8], tgt_line: &mut[u8], _: &[u8], _: &mut[u8],
                                                 chan_convert: LineConverter)
{
    chan_convert(src_line, tgt_line)
}

fn read_idat_stream<R: Reader>(dc: &mut PngDecoder<R>, len: &mut usize, palette: &[u8])
                                                                    -> IoResult<Vec<u8>>
{
    let filter_step = if dc.src_indexed { 1 } else { dc.src_fmt.channels() };
    let tgt_bytespp = dc.tgt_fmt.channels() as usize;
    let tgt_linesize = dc.w as usize * tgt_bytespp;

    let mut result: Vec<u8> = repeat(0).take(dc.w as usize * dc.h as usize * tgt_bytespp).collect();
    let mut depaletted_line: Vec<u8> = if dc.src_indexed {
        repeat(0).take((dc.w * 3) as usize).collect()
    } else {
        Vec::new()
    };

    let chan_convert = match get_converter(dc.src_fmt, dc.tgt_fmt) {
        Some(c) => c,
        None => return IFErr!("no such converter"),
    };

    let convert: fn(&[u8], &mut[u8], &[u8], &mut[u8], LineConverter) = if dc.src_indexed {
        depalette_convert
    } else {
        simple_convert
    };

    try!(fill_uc_buf(dc, len));

    match dc.ilace {
        PngInterlace::None => {
            let src_linesize = dc.w * filter_step;
            let mut cline: Vec<u8> = repeat(0).take(src_linesize+1).collect();
            let mut pline: Vec<u8> = repeat(0).take(src_linesize+1).collect();

            let mut ti = 0us;
            for _j in (0 .. dc.h) {
                next_uncompressed_line(dc, &mut cline[]);
                let filter_type: u8 = cline[0];

                try!(recon(
                    &mut cline[1 .. src_linesize+1], &mut pline[1 .. src_linesize+1],
                    filter_type, filter_step
                ));

                convert(&cline[1..], &mut result[ti .. ti+tgt_linesize], palette, &mut depaletted_line[], chan_convert);

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

            for pass in (0..7us) {
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

                    next_uncompressed_line(dc, &mut cline[]);
                    let filter_type: u8 = cline[0];

                    try!(recon(
                        &mut cline[1 .. src_linesize+1], &pline[1 .. src_linesize+1],
                        filter_type, filter_step
                    ));

                    convert(&cline[1..],
                            &mut redlinebuf[0..redw[pass] * tgt_bytespp],
                            palette,
                            &mut depaletted_line[],
                            chan_convert);

                    let mut redi = 0us;
                    for i in (0 .. redw[pass]) {
                        let tgt = tgt_px(i, j, dc.w) * tgt_bytespp;
                        copy_memory(
                            &mut result[tgt .. tgt+tgt_bytespp],
                            &redlinebuf[redi .. redi+tgt_bytespp]
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

fn fill_uc_buf<R: Reader>(dc: &mut PngDecoder<R>, len: &mut usize) -> IoResult<()> {
    let mut chunks: Vec<Vec<u8>> = Vec::new();
    let mut totallen = 0us;
    loop {
        let mut fresh: Vec<u8> = repeat(0).take(*len).collect();
        try!(dc.stream.read_at_least(*len, &mut fresh[]));
        dc.crc.put(&fresh[]);
        chunks.push(fresh);
        totallen += *len;

        // crc
        try!(dc.stream.read_at_least(4, &mut dc.chunk_crc[0..4]));
        if &dc.crc.finish_be()[] != &dc.chunk_crc[0..4] {
            return IFErr!("corrupt image data");
        }

        // next chunk's len and type
        *len = try!(read_chunkmeta(dc));

        if &dc.chunk_lentype[4..8] != b"IDAT" {
            break;
        }
    }

    let mut alldata: Vec<u8> = repeat(0).take(totallen).collect();
    let mut di = 0us;
    for chunk in chunks.iter() {
        copy_memory(&mut alldata[di .. di+chunk.len()], &chunk[]);
        di += chunk.len();
    }

    let inflated = match inflate_bytes_zlib(&alldata[]) {
        Some(cvec) => cvec,
        None => return IFErr!("could not inflate zlib source")
    };

    dc.uc_buf = repeat(0u8).take(inflated.as_slice().len()).collect();
    copy_memory(&mut dc.uc_buf[], inflated.as_slice());

    Ok(())
}

fn next_uncompressed_line<R: Reader>(dc: &mut PngDecoder<R>, dst: &mut[u8]) {
    let dstlen = dst.len();
    copy_memory(dst, &dc.uc_buf[dc.uc_start .. dc.uc_start + dstlen]);
    dc.uc_start += dst.len();
}

fn recon(cline: &mut[u8], pline: &[u8], ftype: u8, fstep: usize) -> IoResult<()> {
    match FromPrimitive::from_u8(ftype) {
        Some(PngFilter::None)
            => { }
        Some(PngFilter::Sub) => {
            for k in (fstep .. cline.len()) {
                cline[k] += cline[k-fstep];
            }
        }
        Some(PngFilter::Up) => {
            for k in (0 .. cline.len()) {
                cline[k] += pline[k];
            }
        }
        Some(PngFilter::Average) => {
            for k in (0 .. fstep) {
                cline[k] += pline[k] / 2;
            }
            for k in (fstep .. cline.len()) {
                cline[k] +=
                    ((cline[k-fstep] as usize + pline[k] as usize) / 2) as u8;
            }
        }
        Some(PngFilter::Paeth) => {
            for k in (0 .. fstep) {
                cline[k] += paeth(0, pline[k], 0);
            }
            for k in (fstep .. cline.len()) {
                cline[k] += paeth(cline[k-fstep], pline[k], pline[k-fstep]);
            }
        }
        None => return IFErr!("filter type not supported"),
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

#[derive(FromPrimitive)]
enum PngFilter {
    None = 0,
    Sub,
    Up,
    Average,
    Paeth,
}

// --------------------------------------------------
// PNG encoder

pub struct PngCustomChunk {
    pub name: [u8; 4],
    pub data: Vec<u8>,
}

pub fn write_png<W: Writer>(writer: &mut W, w: usize, h: usize, data: &[u8], tgt_fmt: ColFmt)
                                                                              -> IoResult<()>
{
    write_png_chunks(writer, w, h, data, tgt_fmt, &[])
}

pub fn write_png_chunks<W: Writer>(writer: &mut W, w: usize, h: usize, data: &[u8],
                                                                   tgt_fmt: ColFmt,
                                                         chunks: &[PngCustomChunk])
                                                                    -> IoResult<()>
{
    let src_chans = data.len() / w / h;
    if w < 1 || h < 1 || (src_chans * w * h != data.len()) {
        return IFErr!("invalid dimensions or data length");
    }

    let src_fmt = match src_chans {
        1 => ColFmt::Y,
        2 => ColFmt::YA,
        3 => ColFmt::RGB,
        4 => ColFmt::RGBA,
        _ => return IFErr!("format not supported"),
    };

    let tgt_fmt = match tgt_fmt {
        ColFmt::Auto                         => src_fmt,
        ColFmt::Y | ColFmt::YA | ColFmt::RGB | ColFmt::RGBA => tgt_fmt,
        _ => return IFErr!("invalid format"),
    };

    let ec = &mut PngEncoder {
        stream    : writer,
        w         : w,
        h         : h,
        src_chans : src_chans,
        src_fmt   : src_fmt,
        tgt_fmt   : tgt_fmt,
        data      : data,
        crc       : Crc32::new(),
    };

    try!(write_png_header(ec));
    for chunk in chunks.iter() {
        try!(write_custom_chunk(ec, chunk));
    }
    try!(write_png_image_data(ec));

    let iend: &'static[u8] = b"\0\0\0\0IEND\xae\x42\x60\x82";
    ec.stream.write_all(iend)
}

fn write_png_header<W: Writer>(ec: &mut PngEncoder<W>) -> IoResult<()> {
    let mut hdr: [u8; 33] = [0; 33];

    copy_memory(&mut hdr[0..8], &PNG_FILE_HEADER[]);
    copy_memory(&mut hdr[8..16], b"\0\0\0\x0dIHDR");
    copy_memory(&mut hdr[16..20], &u32_to_be(ec.w as u32)[]);
    copy_memory(&mut hdr[20..24], &u32_to_be(ec.h as u32)[]);
    hdr[24] = 8;    // bit depth
    hdr[25] = match ec.tgt_fmt {    // color type
        ColFmt::Y => PngColortype::Y,
        ColFmt::YA => PngColortype::YA,
        ColFmt::RGB => PngColortype::RGB,
        ColFmt::RGBA => PngColortype::RGBA,
        _ => return IFErr!("not supported"),
    } as u8;
    copy_memory(&mut hdr[26..29], &[0, 0, 0]);  // compression, filter, interlace
    ec.crc.put(&hdr[12..29]);
    copy_memory(&mut hdr[29..33], &ec.crc.finish_be()[]);

    ec.stream.write_all(&hdr[])
}

fn write_custom_chunk<W: Writer>(ec: &mut PngEncoder<W>, chunk: &PngCustomChunk) -> IoResult<()> {
    if chunk.name[0] < 97 || 122 < chunk.name[0] { return IFErr!("invalid chunk name"); }
    for b in (&chunk.name[1..]).iter() {
        if *b < 65 || (90 < *b && *b < 97) || 122 < *b {
            return IFErr!("invalid chunk name");
        }
    }
    if 0x7fff_ffff < chunk.data.len() { return IFErr!("chunk too long"); }

    try!(ec.stream.write_be_u32(chunk.data.len() as u32));
    try!(ec.stream.write_all(&chunk.name[]));
    try!(ec.stream.write_all(&chunk.data[]));
    let mut crc = Crc32::new();
    crc.put(&chunk.name[]);
    crc.put(&chunk.data[]);
    ec.stream.write_all(&crc.finish_be()[])
}

struct PngEncoder<'r, W:'r> {
    stream        : &'r mut W,   // TODO is this ok?
    w             : usize,
    h             : usize,
    src_chans     : usize,
    tgt_fmt       : ColFmt,
    src_fmt       : ColFmt,
    data          : &'r [u8],
    crc           : Crc32,
}

fn write_png_image_data<W: Writer>(ec: &mut PngEncoder<W>) -> IoResult<()> {
    let convert = match get_converter(ec.src_fmt, ec.tgt_fmt) {
        Some(c) => c,
        None => return IFErr!("no such converter"),
    };

    let filter_step = ec.tgt_fmt.channels();
    let tgt_linesize = ec.w * filter_step + 1;   // +1 for filter type
    let mut cline: Vec<u8> = repeat(0).take(tgt_linesize).collect();
    let mut pline: Vec<u8> = repeat(0).take(tgt_linesize).collect();
    let mut filtered_image: Vec<u8> = repeat(0).take(tgt_linesize * ec.h).collect();

    let src_linesize = ec.w * ec.src_chans;

    let mut ti = 0us;
    for si in range_step(0, ec.h * src_linesize, src_linesize) {
        convert(&ec.data[si .. si+src_linesize], &mut cline[1 .. tgt_linesize]);

        for i in (1 .. filter_step+1) {
            filtered_image[ti+i] = cline[i] - paeth(0, pline[i], 0)
        }
        for i in (filter_step+1 .. cline.len()) {
            filtered_image[ti+i] =
                cline[i] - paeth(cline[i-filter_step], pline[i], pline[i-filter_step])
        }

        filtered_image[ti] = PngFilter::Paeth as u8;

        let swap = pline;
        pline = cline;
        cline = swap;

        ti += tgt_linesize;
    }

    let compressed = match deflate_bytes_zlib(&filtered_image[]) {
        Some(cvec) => cvec,
        None => return IFErr!("compression failed"),
    };
    ec.crc.put(b"IDAT");
    ec.crc.put(compressed.as_slice());
    let crc = &ec.crc.finish_be();

    // TODO split up data into smaller chunks?
    let chunklen = compressed.as_slice().len() as u32;
    try!(ec.stream.write_all(&u32_to_be(chunklen)[]));
    try!(ec.stream.write_all(b"IDAT"));
    try!(ec.stream.write_all(compressed.as_slice()));
    try!(ec.stream.write_all(crc));
    Ok(())
}

// ------------------------------------------------------------

#[derive(Debug)]
pub struct TgaHeader {
   pub id_length      : u8,
   pub palette_type   : u8,
   pub data_type      : u8,
   pub palette_start  : u16,
   pub palette_length : u16,
   pub palette_bits   : u8,
   pub x_origin       : u16,
   pub y_origin       : u16,
   pub width          : u16,
   pub height         : u16,
   pub bits_pp        : u8,
   pub flags          : u8,
}

/** Returns: basic info. NOTE: Even if the file contains BGR/A data it's reported as
 * RGB/A. Returns error if data type is not supported by the decoder, at least
 * for now (don't rely on it, it's not that way on purpose).
 */
pub fn read_tga_info<R: Reader>(reader: &mut R) -> IoResult<IFInfo> {
    use self::ColFmt::*;

    let hdr = try!(read_tga_header(reader));
    let TgaInfo { src_chans, src_fmt, rle } = try!(parse_tga_header(&hdr));
    let _src_chans = src_chans; let _rle = rle; // warnings be gone

    let reported_fmt = match src_fmt {
        ColFmt::Y => ColFmt::Y,
        ColFmt::YA => ColFmt::YA,
        ColFmt::BGR => ColFmt::RGB,
        ColFmt::BGRA => ColFmt::RGBA,
        _ => return IFErr!("source format unknown"),
    };

    Ok(IFInfo {
        w: hdr.width as usize,
        h: hdr.height as usize,
        c: reported_fmt,
    })
}

pub fn read_tga_header<R: Reader>(reader: &mut R) -> IoResult<TgaHeader> {
    let mut buf = [0u8; 18];
    try!(reader.read_at_least(buf.len(), &mut buf));

    Ok(TgaHeader {
        id_length      : buf[0],
        palette_type   : buf[1],
        data_type      : buf[2],
        palette_start  : u16_from_le(&buf[3..5]),
        palette_length : u16_from_le(&buf[5..7]),
        palette_bits   : buf[7],
        x_origin       : u16_from_le(&buf[8..10]),
        y_origin       : u16_from_le(&buf[10..12]),
        width          : u16_from_le(&buf[12..14]),
        height         : u16_from_le(&buf[14..16]),
        bits_pp        : buf[16],
        flags          : buf[17],
    })
}

pub fn read_tga<R: Reader>(reader: &mut R, req_fmt: ColFmt) -> IoResult<IFImage> {
    let hdr = try!(read_tga_header(reader));

    if 0 < hdr.palette_type { return IFErr!("paletted TGAs not supported"); }
    if hdr.width < 1 || hdr.height < 1 { return IFErr!("invalid dimensions"); }
    if 0 < (hdr.flags & 0xc0) { return IFErr!("interlaced TGAs not supported"); }
    if 0 < (hdr.flags & 0x10) { return IFErr!("right-to-left TGAs not supported"); }

    let TgaInfo { src_chans, src_fmt, rle } = try!(parse_tga_header(&hdr));

    let tgt_fmt = {
        use self::ColFmt::*;
        match req_fmt {
            Y | YA | RGB | RGBA => req_fmt,
            Auto => match src_fmt {
                Y => Y,
                YA => YA,
                BGR => RGB,
                BGRA => RGBA,
                _ => return IFErr!("not supported"),
            },
            _ => return IFErr!("conversion not supported"),
        }
    };

    try!(skip(reader, hdr.id_length as usize));

    let dc = &mut TgaDecoder {
        stream         : reader,
        w              : hdr.width as usize,
        h              : hdr.height as usize,
        origin_at_top  : 0 < (hdr.flags & 0x20),
        src_chans      : src_chans,
        rle            : rle,
        src_fmt        : src_fmt,
        tgt_fmt        : tgt_fmt,
    };

    Ok(IFImage {
        w      : dc.w,
        h      : dc.h,
        c      : dc.tgt_fmt,
        pixels : try!(decode_tga(dc))
    })
}

struct TgaInfo {
    src_chans : usize,
    src_fmt   : ColFmt,
    rle       : bool,
}

// Returns: source color format and whether it's RLE compressed
fn parse_tga_header(hdr: &TgaHeader) -> IoResult<TgaInfo> {
    use self::TgaDataType::*;

    let mut rle = false;
    let datatype: Option<TgaDataType> = FromPrimitive::from_u8(hdr.data_type);
    let attr_bits_pp = hdr.flags & 0xf;
    match (datatype, hdr.bits_pp, attr_bits_pp) {
        (Some(TrueColor),    24, 0) => {}
        (Some(TrueColor),    32, 8) => {}
        (Some(TrueColor),    32, 0) => {} // some pics say 0 bits for attr although 8 bits present
        (Some(Gray),          8, 0) => {}
        (Some(Gray),         16, 8) => {}
        (Some(TrueColorRLE), 24, 0) => { rle = true; }
        (Some(TrueColorRLE), 32, 8) => { rle = true; }
        (Some(TrueColorRLE), 32, 0) => { rle = true; }  // again, 8 bits are present in some pics
        (Some(GrayRLE),       8, 0) => { rle = true; }
        (Some(GrayRLE),      16, 8) => { rle = true; }
        _ => return IFErr!("data type not supported")
    }

    let src_chans = hdr.bits_pp as usize / 8;
    let src_fmt = match src_chans {
        1 => ColFmt::Y,
        2 => ColFmt::YA,
        3 => ColFmt::BGR,
        4 => ColFmt::BGRA,
        _ => return IFErr!("not supported"),
    };

    Ok(TgaInfo {
        src_chans : src_chans,
        src_fmt   : src_fmt,
        rle       : rle,
    })
}

fn decode_tga<R: Reader>(dc: &mut TgaDecoder<R>) -> IoResult<Vec<u8>> {
    let tgt_chans = dc.tgt_fmt.channels();
    let tgt_linesize = (dc.w * tgt_chans) as i64;
    let src_linesize = (dc.w * dc.src_chans) as usize;

    let mut src_line: Vec<u8> = repeat(0).take(src_linesize).collect();
    let mut result: Vec<u8> = repeat(0).take((dc.w * dc.h * tgt_chans) as usize).collect();

    let (tgt_stride, mut ti): (i64, i64) =
        if dc.origin_at_top {
            (tgt_linesize, 0)
        } else {
            (-tgt_linesize, (dc.h-1) as i64 * tgt_linesize)
        };

    let convert: LineConverter = match get_converter(dc.src_fmt, dc.tgt_fmt) {
        Some(c) => c,
        None => return IFErr!("no such converter"),
    };

    if !dc.rle {
        for _j in (0 .. dc.h) {
            try!(dc.stream.read_at_least(src_linesize, &mut src_line[]));
            convert(&src_line[], &mut result[ti as usize..(ti+tgt_linesize) as usize]);
            ti += tgt_stride;
        }
        return Ok(result);
    }

    // ---- RLE ----

    let bytes_pp = dc.src_chans as usize;
    let mut rbuf: Vec<u8> = repeat(0u8).take(src_linesize).collect();
    let mut plen = 0us;    // packet length
    let mut its_rle = false;

    for _ in (0 .. dc.h) {
        // fill src_line with uncompressed data
        let mut wanted: usize = src_linesize;
        while 0 < wanted {
            if plen == 0 {
                let hdr = try!(dc.stream.read_u8()) as usize;
                its_rle = 0 < (hdr & 0x80);
                plen = ((hdr & 0x7f) + 1) * bytes_pp;
            }
            let gotten: usize = src_linesize - wanted;
            let copysize: usize = min(plen, wanted);
            if its_rle {
                try!(dc.stream.read_at_least(bytes_pp, &mut rbuf[0..bytes_pp]));
                for p in range_step(gotten, gotten+copysize, bytes_pp) {
                    copy_memory(&mut src_line[p..p+bytes_pp], &rbuf[0..bytes_pp]);
                }
            } else {    // it's raw
                let slice: &mut[u8] = &mut src_line[gotten..gotten+copysize];
                try!(dc.stream.read_at_least(copysize, slice));
            }
            wanted -= copysize;
            plen -= copysize;
        }

        convert(&src_line[], &mut result[ti as usize .. (ti+tgt_linesize) as usize]);
        ti += tgt_stride;
    }

    Ok(result)
}

struct TgaDecoder<'r, R:'r> {
    stream        : &'r mut R,   // TODO is this ok?
    w             : usize,
    h             : usize,
    origin_at_top : bool,    // src
    src_chans     : usize,
    rle           : bool,          // run length compressed
    src_fmt       : ColFmt,
    tgt_fmt       : ColFmt,
}

#[derive(FromPrimitive)]
enum TgaDataType {
    Idx          = 1,
    TrueColor    = 2,
    Gray         = 3,
    IdxRLE       = 9,
    TrueColorRLE = 10,
    GrayRLE      = 11,
}

// --------------------------------------------------
// TGA Encoder

/// For tgt_fmt, accepts FmtRGB/A but not FmtBGR/A; will encode as BGR/A.
pub fn write_tga<W: Writer>(writer: &mut W, w: usize, h: usize, data: &[u8], tgt_fmt: ColFmt)
                                                                              -> IoResult<()>
{
    if w < 1 || h < 1 || 0xffff < w || 0xffff < h {
        return IFErr!("invalid dimensions");
    }

    let src_chans = data.len() / w / h;
    if src_chans * w * h != data.len() {
        return IFErr!("mismatching dimensions and length");
    }

    let src_fmt = match src_chans {
        1 => ColFmt::Y,
        2 => ColFmt::YA,
        3 => ColFmt::RGB,
        4 => ColFmt::RGBA,
        _ => return IFErr!("format not supported"),
    };

    let tgt_fmt = {
        use self::ColFmt::*;
        match (src_fmt, tgt_fmt) {
            (Y, Auto) | (YA, Auto) => src_fmt,
            (RGB, Auto)            => BGR,
            (RGBA, Auto)           => BGRA,
            (_, Y) | (_, YA)       => tgt_fmt,
            (_, RGB)               => BGR,
            (_, RGBA)              => BGRA,
            _ => return IFErr!("invalid format"),
        }
    };

    let ec = &mut TgaEncoder {
        stream    : writer,
        w         : w,
        h         : h,
        src_chans : src_chans,
        tgt_chans : tgt_fmt.channels(),
        src_fmt   : src_fmt,
        tgt_fmt   : tgt_fmt,
        rle       : true,
        data      : data,
    };

    try!(write_tga_header(ec));
    try!(write_tga_image_data(ec));

    // write footer
    let ftr: &'static [u8] =
        b"\x00\x00\x00\x00\
          \x00\x00\x00\x00\
          TRUEVISION-XFILE.\x00";
    try!(ec.stream.write_all(ftr));

    ec.stream.flush()
}

fn write_tga_header<W: Writer>(ec: &mut TgaEncoder<W>) -> IoResult<()> {
    use self::TgaDataType::*;
    let (data_type, has_alpha) = match ec.tgt_chans {
        1 => (if ec.rle { GrayRLE      } else { Gray      }, false),
        2 => (if ec.rle { GrayRLE      } else { Gray      }, true),
        3 => (if ec.rle { TrueColorRLE } else { TrueColor }, false),
        4 => (if ec.rle { TrueColorRLE } else { TrueColor }, true),
        _ => return IFErr!("internal error")
    };

    let w = u16_to_le(ec.w as u16);
    let h = u16_to_le(ec.h as u16);
    let hdr: &[u8; 18] = &[
        0, 0,
        data_type as u8,
        0, 0, 0, 0, 0,
        0, 0, 0, 0,
        w[0], w[1],
        h[0], h[1],
        (ec.tgt_chans * 8) as u8,
        if has_alpha {8u8} else {0u8},  // flags
    ];

    ec.stream.write_all(hdr)
}

fn write_tga_image_data<W: Writer>(ec: &mut TgaEncoder<W>) -> IoResult<()> {
    let src_linesize = (ec.w * ec.src_chans) as usize;
    let tgt_linesize = (ec.w * ec.tgt_chans) as usize;
    let mut tgt_line: Vec<u8> = repeat(0u8).take(tgt_linesize).collect();
    let mut si = (ec.h-1) as usize * src_linesize;

    let convert = match get_converter(ec.src_fmt, ec.tgt_fmt) {
        Some(c) => c,
        None => return IFErr!("no such converter"),
    };

    if !ec.rle {
        for _ in (0 .. ec.h) {
            convert(&ec.data[si..si+src_linesize], &mut tgt_line[]);
            try!(ec.stream.write_all(&tgt_line[]));
            si -= src_linesize; // origin at bottom
        }
        return Ok(());
    }

    // ----- RLE -----

    let bytes_pp = ec.tgt_chans as usize;
    let max_packets_per_line = (tgt_linesize+127) / 128;
    let mut cmp_buf: Vec<u8> = repeat(0u8).take(tgt_linesize+max_packets_per_line).collect();
    for _ in (0 .. ec.h) {
        convert(&ec.data[si .. si+src_linesize], &mut tgt_line[]);
        let compressed_line = rle_compress(&tgt_line[], &mut cmp_buf[], ec.w, bytes_pp);
        try!(ec.stream.write_all(&compressed_line[]));
        si -= src_linesize;
    }
    return Ok(());
}

fn rle_compress<'a>(line: &[u8], cmp_buf: &'a mut[u8], w: usize, bytes_pp: usize)
                                                                    -> &'a [u8]
{
    let rle_limit = if 1 < bytes_pp { 2 } else { 3 };   // run len worth an RLE packet
    let mut rawlen = 0us;
    let mut raw_i = 0us;   // start of raw packet data
    let mut cmp_i = 0us;
    let mut pixels_left = w;
    let mut px: &[u8];

    let mut i = bytes_pp;
    while 0 < pixels_left {
        let mut runlen = 1us;
        px = &line[i-bytes_pp .. i];
        while i < line.len() && px == &line[i..i+bytes_pp] && runlen < 128 {
            runlen += 1;
            i += bytes_pp;
        }
        pixels_left -= runlen;

        if runlen < rle_limit {
            // data goes to raw packet
            rawlen += runlen;
            if 128 <= rawlen {  // full packet, need to store it
                let copysize = 128 * bytes_pp;
                cmp_buf[cmp_i] = 0x7f; cmp_i += 1;  // packet header
                copy_memory(
                    &mut cmp_buf[cmp_i..cmp_i+copysize],
                    &line[raw_i..raw_i+copysize]
                );
                cmp_i += copysize;
                raw_i += copysize;
                rawlen -= 128;
            }
        } else {
            // RLE packet is worth it
            // store raw packet first, when needed
            if 0 < rawlen {
                let copysize = rawlen * bytes_pp;
                cmp_buf[cmp_i] = (rawlen-1) as u8;    // packet header
                cmp_i += 1;
                copy_memory(
                    &mut cmp_buf[cmp_i..cmp_i+copysize],
                    &line[raw_i..raw_i+copysize]
                );
                cmp_i += copysize;
                rawlen = 0;
            }

            // store RLE packet
            cmp_buf[cmp_i] = (0x80 | (runlen-1)) as u8;   // packet header
            cmp_i += 1;
            copy_memory(
                &mut cmp_buf[cmp_i..cmp_i+bytes_pp],
                &px[0..bytes_pp]
            );
            cmp_i += bytes_pp;
            raw_i = i;
        }
        i += bytes_pp;
    }   // for

    if 0 < rawlen {     // last packet of the line, if any
        let copysize = rawlen * bytes_pp;
        cmp_buf[cmp_i] = (rawlen-1) as u8;    // packet header
        cmp_i += 1;
        copy_memory(
            &mut cmp_buf[cmp_i..cmp_i+copysize],
            &line[raw_i..raw_i+copysize]
        );
        cmp_i += copysize;
    }

    &cmp_buf[0 .. cmp_i]
}

struct TgaEncoder<'r, R:'r> {
    stream        : &'r mut R,   // TODO is this ok?
    w             : usize,
    h             : usize,
    src_chans     : usize,
    tgt_chans     : usize,
    tgt_fmt       : ColFmt,
    src_fmt       : ColFmt,
    rle           : bool,          // run length compressed
    data          : &'r [u8],
}

// ------------------------------------------------------------
// Baseline JPEG decoder

pub fn read_jpeg_info<R: Reader>(stream: &mut R) -> IoResult<IFInfo> {
    try!(read_jfif(stream));

    loop {
        let mut marker: [u8; 2] = [0, 0];
        try!(stream.read_at_least(marker.len(), &mut marker[]));

        if marker[0] != 0xff { return IFErr!("no marker"); }
        while marker[1] == 0xff {
            try!(stream.read_at_least(1, &mut marker[1..2]));
        }

        match marker[1] {
            SOF0 | SOF2 => {
                let mut tmp: [u8; 8] = [0,0,0,0, 0,0,0,0];
                try!(stream.read_at_least(tmp.len(), &mut tmp[]));
                return Ok(IFInfo {
                    w: u16_from_be(&tmp[5..7]) as usize,
                    h: u16_from_be(&tmp[3..5]) as usize,
                    c: match tmp[7] {
                           1 => ColFmt::Y,
                           3 => ColFmt::RGB,
                           _ => return IFErr!("not valid baseline jpeg")
                       },
                });
            }
            SOS | EOI => return IFErr!("no frame header"),
            DRI | DHT | DQT | COM | APP0 ... APPF => {
                let mut tmp: [u8; 2] = [0, 0];
                try!(stream.read_at_least(tmp.len(), &mut tmp[]));
                let len = u16_from_be(&mut tmp[]) - 2;
                try!(skip(stream, len as usize));
            }
            _ => return IFErr!("invalid / unsupported marker"),
        }
    }
}

pub fn read_jpeg<R: Reader>(reader: &mut R, req_fmt: ColFmt) -> IoResult<IFImage> {
    use self::ColFmt::*;
    let req_fmt = match req_fmt {
        Auto | Y | YA | RGB | RGBA => req_fmt,
        _ => return IFErr!("format not supported")
    };

    try!(read_jfif(reader));

    let dc = &mut JpegDecoder {
        stream      : reader,
        w           : 0,
        h           : 0,
        tgt_fmt     : req_fmt,
        eoi_reached : false,
        has_frame_header : false,
        qtables     : [[0; 64]; 4],
        ac_tables   : unsafe { zeroed() },
        dc_tables   : unsafe { zeroed() },
        cb          : 0,
        bits_left   : 0,
        num_mcu_x   : 0,
        num_mcu_y   : 0,
        restart_interval : 0,
        comps       : unsafe { zeroed() },
        index_for   : [0, 0, 0],
        num_comps   : 0,
        hmax        : 0,
        vmax        : 0,
    };

    try!(read_markers(dc));   // reads until first scan header

    if dc.eoi_reached {
        return IFErr!("no image data");
    }
    dc.tgt_fmt =
        if req_fmt == ColFmt::Auto {
            match dc.num_comps {
                1 => ColFmt::Y, 3 => ColFmt::RGB,
                _ => return IFErr!("internal error")
            }
        } else {
            req_fmt
        };

    for comp in dc.comps.iter_mut() {
        comp.data = repeat(0u8).take(dc.num_mcu_x*comp.sfx*8*dc.num_mcu_y*comp.sfy*8).collect();
    }

    Ok(IFImage {
        w      : dc.w,
        h      : dc.h,
        c      : dc.tgt_fmt,
        pixels : {
            // progressive images aren't supported so only one scan
            try!(decode_scan(dc));
            // throw away fill samples and convert to target format
            try!(reconstruct(dc))
        }
    })
}

fn read_jfif<R: Reader>(reader: &mut R) -> IoResult<()> {
    let mut buf = [0u8; 20]; // SOI, APP0
    try!(reader.read_at_least(buf.len(), &mut buf[]));

    let len = u16_from_be(&buf[4..6]) as usize;

    if &buf[0..4] != &[0xff_u8, 0xd8, 0xff, 0xe0][] ||
       &buf[6..11] != b"JFIF\0" || len < 16 {
        return IFErr!("not JPEG/JFIF");
    }

    if buf[11] != 1 {
        return IFErr!("version not supported");
    }

    // ignore density_unit, -x, -y at 13, 14..16, 16..18

    let thumbsize = buf[18] as usize * buf[19] as usize * 3;
    if thumbsize != len - 16 {
        return IFErr!("corrupt jpeg header");
    }
    skip(reader, thumbsize)
}

struct JpegDecoder<'r, R: Reader + 'r> {
    stream        : &'r mut R,
    w             : usize,
    h             : usize,
    tgt_fmt       : ColFmt,

    eoi_reached      : bool,
    has_frame_header : bool,

    qtables     : [[u8; 64]; 4],
    ac_tables   : [HuffTab; 2],
    dc_tables   : [HuffTab; 2],

    cb          : u8,   // current byte
    bits_left   : usize, // num of unused bits in cb

    num_mcu_x   : usize,
    num_mcu_y   : usize,
    restart_interval : usize,

    comps       : [Component; 3],
    index_for   : [usize; 3],
    num_comps   : usize,
    hmax        : usize,
    vmax        : usize,
}

struct HuffTab {
    values  : [u8; 256],
    sizes   : [u8; 257],
    mincode : [i16; 16],
    maxcode : [i16; 16],
    valptr  : [i16; 16],
}

struct Component {
    id       : u8,
    sfx      : usize,            // sampling factor, aka. h
    sfy      : usize,            // sampling factor, aka. v
    x        : usize,          // total number of samples without fill samples
    y        : usize,          // total number of samples without fill samples
    qtable   : usize,
    ac_table : usize,
    dc_table : usize,
    pred     : isize,          // dc prediction
    data     : Vec<u8>,      // reconstructed samples
}

fn read_markers<R: Reader>(dc: &mut JpegDecoder<R>) -> IoResult<()> {
    let mut has_next_scan_header = false;
    while !has_next_scan_header && !dc.eoi_reached {
        let mut marker: [u8; 2] = [0, 0];
        try!(dc.stream.read_at_least(marker.len(), &mut marker[]));

        if marker[0] != 0xff { return IFErr!("no marker"); }
        while marker[1] == 0xff {
            try!(dc.stream.read_at_least(1, &mut marker[1..2]));
        }

        //println!("marker: 0x{:x}", marker[1]);
        match marker[1] {
            DHT => try!(read_huffman_tables(dc)),
            DQT => try!(read_quantization_tables(dc)),
            SOF0 => {
                if dc.has_frame_header {
                    return IFErr!("extra frame header");
                }
                try!(read_frame_header(dc));
                dc.has_frame_header = true;
            }
            SOS => {
                if !dc.has_frame_header {
                    return IFErr!("no frame header");
                }
                try!(read_scan_header(dc));
                has_next_scan_header = true;
            }
            DRI => try!(read_restart_interval(dc)),
            EOI => dc.eoi_reached = true,
            APP0 ... APPF | COM => {
                //println!("skipping unknown marker...");
                let mut tmp: [u8; 2] = [0, 0];
                try!(dc.stream.read_at_least(tmp.len(), &mut tmp[]));
                let len = u16_from_be(&mut tmp[]) - 2;
                try!(skip(dc.stream, len as usize));
            }
            _ => return IFErr!("invalid / unsupported marker"),
        }
    }
    Ok(())
}

//const SOI: u8 = 0xd8;     // start of image
const SOF0: u8 = 0xc0;    // start of frame / baseline DCT
//const SOF1: u8 = 0xc1;    // start of frame / extended seq.
const SOF2: u8 = 0xc2;    // start of frame / progressive DCT
//const SOF3: u8 = 0xc3;    // start of frame / lossless
//const SOF9: u8 = 0xc9;    // start of frame / extended seq., arithmetic
//const SOF11: u8 = 0xcb;    // start of frame / lossless, arithmetic
const DHT: u8 = 0xc4;     // define huffman tables
const DQT: u8 = 0xdb;     // define quantization tables
const DRI: u8 = 0xdd;     // define restart interval
const SOS: u8 = 0xda;     // start of scan
//const DNL: u8 = 0xdc;     // define number of lines
const RST0: u8 = 0xd0;    // restart entropy coded data
// ...
const RST7: u8 = 0xd7;    // restart entropy coded data
const APP0: u8 = 0xe0;    // application 0 segment
// ...
const APPF: u8 = 0xef;    // application f segment
//const DAC: u8 = 0xcc;     // define arithmetic conditioning table
const COM: u8 = 0xfe;     // comment
const EOI: u8 = 0xd9;     // end of image

fn read_huffman_tables<R: Reader>(dc: &mut JpegDecoder<R>) -> IoResult<()> {
    let mut buf = [0u8; 17];
    try!(dc.stream.read_at_least(2, &mut buf[0..2]));
    let mut len = u16_from_be(&buf[0..2]) as isize -2;

    while 0 < len {
        try!(dc.stream.read_at_least(17, &mut buf[0..17]));  // info byte and BITS
        let table_class = buf[0] >> 4;            // 0 = dc table, 1 = ac table
        let table_slot = (buf[0] & 0xf) as usize;  // must be 0 or 1 for baseline
        if 1 < table_slot || 1 < table_class {
            return IFErr!("invalid / not supported");
        }

        // compute total number of huffman codes
        let mut mt = 0us;
        for i in (1..17us) {
            mt += buf[i] as usize;
        }
        if 256 < mt {
            return IFErr!("invalid / not supported");
        }

        if table_class == 0 {
            try!(dc.stream.read_at_least(mt, &mut dc.dc_tables[table_slot].values[0..mt]));
            derive_table(&mut dc.dc_tables[table_slot], &buf[1..17]);
        } else {
            try!(dc.stream.read_at_least(mt, &mut dc.ac_tables[table_slot].values[0..mt]));
            derive_table(&mut dc.ac_tables[table_slot], &buf[1..17]);
        }

        len -= 17 + mt as isize;
    }
    Ok(())
}

fn derive_table(table: &mut HuffTab, num_values: &[u8]) {
    assert!(num_values.len() == 16);

    let mut codes: [i16; 256] = [0; 256];

    let mut k = 0;
    for i in (0..16us) {
        for _j in (0 .. num_values[i]) {
            table.sizes[k] = (i + 1) as u8;
            k += 1;
        }
    }
    table.sizes[k] = 0;

    k = 0;
    let mut code = 0_i16;
    let mut si = table.sizes[k];
    loop {
        while si == table.sizes[k] {
            codes[k] = code;
            code += 1;
            k += 1;
        }

        if table.sizes[k] == 0 { break; }

        while si != table.sizes[k] {
            code <<= 1;
            si += 1;
        }
    }

    derive_mincode_maxcode_valptr(
        &mut table.mincode, &mut table.maxcode, &mut table.valptr,
        &codes, num_values
    );
}

fn derive_mincode_maxcode_valptr(mincode: &mut[i16; 16], maxcode: &mut[i16; 16],
                                 valptr:  &mut[i16; 16], codes: &[i16; 256],
                                 num_values: &[u8])
{
    for i in (0..16us) {
        mincode[i] = -1;
        maxcode[i] = -1;
        valptr[i] = -1;
    }

    let mut j = 0us;
    for i in (0..16us) {
        if num_values[i] != 0 {
            valptr[i] = j as i16;
            mincode[i] = codes[j];
            j += (num_values[i] - 1) as usize;
            maxcode[i] = codes[j];
            j += 1;
        }
    }
}

fn read_quantization_tables<R: Reader>(dc: &mut JpegDecoder<R>) -> IoResult<()> {
    let mut buf = [0u8; 2];
    try!(dc.stream.read_at_least(2, &mut buf[0..2]));
    let mut len = u16_from_be(&buf[0..2]) as usize -2;
    if len % 65 != 0 {
        return IFErr!("invalid / not supported");
    }

    while 0 < len {
        try!(dc.stream.read_at_least(1, &mut buf[0..1]));
        let precision = buf[0] >> 4;  // 0 = 8 bit, 1 = 16 bit
        let table_slot = (buf[0] & 0xf) as usize;
        if 3 < table_slot || precision != 0 {   // only 8 bit for baseline
            return IFErr!("invalid / not supported");
        }
        try!(dc.stream.read_at_least(64, &mut dc.qtables[table_slot]));
        len -= 65;
    }
    Ok(())
}

fn read_frame_header<R: Reader>(dc: &mut JpegDecoder<R>) -> IoResult<()> {
    let mut buf = [0u8; 9];
    try!(dc.stream.read_at_least(8, &mut buf[0..8]));
    let len = u16_from_be(&buf[0..2]) as usize;
    let precision = buf[2];
    dc.h = u16_from_be(&buf[3..5]) as usize;
    dc.w = u16_from_be(&buf[5..7]) as usize;
    dc.num_comps = buf[7] as usize;

    if precision != 8 || (dc.num_comps != 1 && dc.num_comps != 3) ||
       len != 8 + dc.num_comps*3 {
        return IFErr!("invalid / not supported");
    }

    dc.hmax = 0;
    dc.vmax = 0;
    let mut mcu_du = 0; // data units in one mcu
    try!(dc.stream.read_at_least(dc.num_comps*3, &mut buf[0..dc.num_comps*3]));

    for i in (0 .. dc.num_comps) {
        let ci = (buf[i*3]-1) as usize;
        if dc.num_comps <= ci {
            return IFErr!("invalid / not supported");
        }
        dc.index_for[i] = ci;
        let sampling_factors = buf[i*3 + 1];
        let comp = &mut dc.comps[ci];
        *comp = Component {
            id      : buf[i*3],
            sfx     : (sampling_factors >> 4) as usize,
            sfy     : (sampling_factors & 0xf) as usize,
            x       : 0,
            y       : 0,
            qtable  : buf[i*3 + 2] as usize,
            ac_table : 0,
            dc_table : 0,
            pred    : 0,
            data    : Vec::<u8>::new(),
        };
        if comp.sfy < 1 || 4 < comp.sfy ||
           comp.sfx < 1 || 4 < comp.sfx ||
           3 < comp.qtable {
            return IFErr!("invalid / not supported");
        }

        if dc.hmax < comp.sfx { dc.hmax = comp.sfx; }
        if dc.vmax < comp.sfy { dc.vmax = comp.sfy; }
        mcu_du += comp.sfx * comp.sfy;
    }
    if 10 < mcu_du { return IFErr!("invalid / not supported"); }

    for i in (0 .. dc.num_comps) {
        dc.comps[i].x = (dc.w as f64 * dc.comps[i].sfx as f64 / dc.hmax as f64).ceil() as usize;
        dc.comps[i].y = (dc.h as f64 * dc.comps[i].sfy as f64 / dc.vmax as f64).ceil() as usize;
    }

    let mcu_w = dc.hmax * 8;
    let mcu_h = dc.vmax * 8;
    dc.num_mcu_x = (dc.w + mcu_w-1) / mcu_w;
    dc.num_mcu_y = (dc.h + mcu_h-1) / mcu_h;

    Ok(())
}

fn read_scan_header<R: Reader>(dc: &mut JpegDecoder<R>) -> IoResult<()> {
    let mut buf: [u8; 3] = [0, 0, 0];
    try!(dc.stream.read_at_least(buf.len(), &mut buf[]));
    let len = u16_from_be(&buf[0..2]) as usize;
    let num_scan_comps = buf[2] as usize;

    if num_scan_comps != dc.num_comps || len != (6+num_scan_comps*2) {
        return IFErr!("invalid / not supported");
    }

    let mut compbuf: Vec<u8> = repeat(0u8).take(len-3).collect();
    try!(dc.stream.read_at_least(compbuf.len(), &mut compbuf[]));

    for i in (0 .. num_scan_comps) {
        let comp_id = compbuf[i*2];
        let mut ci = 0;
        while ci < dc.num_comps && dc.comps[ci].id != comp_id { ci+=1 }
        if dc.num_comps <= ci {
            return IFErr!("invalid / not supported");
        }

        let tables = compbuf[i*2+1];
        dc.comps[ci].dc_table = (tables >> 4) as usize;
        dc.comps[ci].ac_table = (tables & 0xf) as usize;
        if 1 < dc.comps[ci].dc_table || 1 < dc.comps[i].ac_table {
            return IFErr!("invalid / not supported");
        }
    }

    // ignore last 3 bytes: spectral_start, spectral_end, approx
    Ok(())
}

fn read_restart_interval<R: Reader>(dc: &mut JpegDecoder<R>) -> IoResult<()> {
    let mut buf: [u8; 4] = [0, 0, 0, 0];
    try!(dc.stream.read_at_least(buf.len(), &mut buf[]));
    let len = u16_from_be(&buf[0..2]) as usize;
    if len != 4 { return IFErr!("invalid / not supported"); }
    dc.restart_interval = u16_from_be(&buf[2..4]) as usize;
    Ok(())
}

fn decode_scan<R: Reader>(dc: &mut JpegDecoder<R>) -> IoResult<()> {
    let (mut intervals, mut mcus) =
        if 0 < dc.restart_interval {
            let total_mcus = dc.num_mcu_x * dc.num_mcu_y;
            let ivals = (total_mcus + dc.restart_interval-1) / dc.restart_interval;
            (ivals, dc.restart_interval)
        } else {
            (1, dc.num_mcu_x * dc.num_mcu_y)
        };

    for mcu_j in (0 .. dc.num_mcu_y) {
        for mcu_i in (0 .. dc.num_mcu_x) {

            // decode mcu
            for c in (0 .. dc.num_comps) {
                let comp_idx = dc.index_for[c];
                let comp_sfx = dc.comps[comp_idx].sfx;
                let comp_sfy = dc.comps[comp_idx].sfy;
                let comp_qtab = dc.comps[comp_idx].qtable;

                for du_j in (0 .. comp_sfy) {
                    for du_i in (0 .. comp_sfx) {
                        // decode entropy, dequantize & dezigzag
                        //let data = try!(decode_block(dc, comp, &dc.qtables[comp.qtable]));
                        let data = try!(decode_block(dc, comp_idx, comp_qtab));

                        // idct & level-shift
                        let outx = (mcu_i * comp_sfx + du_i) * 8;
                        let outy = (mcu_j * comp_sfy + du_j) * 8;
                        let dst_stride = dc.num_mcu_x * comp_sfx * 8;
                        let base = &mut dc.comps[comp_idx].data[0] as *mut u8;
                        let offset = (outy * dst_stride + outx) as isize;
                        unsafe {
                            let dst = base.offset(offset);
                            stbi_idct_block(dst, dst_stride, &data[]);
                        }
                    }
                }
            }

            mcus -= 1;

            if mcus == 0 {
                intervals -= 1;
                if intervals == 0 {
                    return Ok(());
                }

                try!(read_restart(dc.stream));    // RSTx marker

                if intervals == 1 {
                    mcus = (dc.num_mcu_y-mcu_j-1) * dc.num_mcu_x + dc.num_mcu_x-mcu_i-1;
                } else {
                    mcus = dc.restart_interval;
                }

                // reset decoder
                dc.cb = 0;
                dc.bits_left = 0;
                for k in (0 .. dc.num_comps) {
                    dc.comps[k].pred = 0;
                }
            }
        }
    }
    Ok(())
}

fn read_restart<R: Reader>(stream: &mut R) -> IoResult<()> {
    let mut buf: [u8; 2] = [0, 0];
    try!(stream.read_at_least(buf.len(), &mut buf[]));
    if buf[0] != 0xff || buf[1] < RST0 || RST7 < buf[1] {
        return IFErr!("reset marker missing");
    }
    Ok(())
}

static DEZIGZAG: [u8; 64] = [
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
];

// decode entropy, dequantize & dezigzag (section F.2)
//fn decode_block<R: Reader>(dc: &mut JpegDecoder<R>, comp: &mut Component,
//                                                     qtable: &[u8, ..64])
//                                                 -> IoResult<[i16, ..64]>
fn decode_block<R: Reader>(dc: &mut JpegDecoder<R>, comp_idx: usize, qtable_idx: usize)
                                                             -> IoResult<[i16; 64]>
{
    //let comp = &mut dc.comps[comp_idx];
    //let qtable = &dc.qtables[qtable_idx];

    let mut res: [i16; 64] = [0; 64];
    //let t = try!(decode_huff(dc, dc.dc_tables[comp.dc_table]));
    let dc_table_idx = dc.comps[comp_idx].dc_table;
    let ac_table_idx = dc.comps[comp_idx].ac_table;
    let t = try!(decode_huff(dc, dc_table_idx, true));
    let diff: isize = if 0 < t { try!(receive_and_extend(dc, t)) } else { 0 };

    dc.comps[comp_idx].pred += diff;
    res[0] = (dc.comps[comp_idx].pred * dc.qtables[qtable_idx][0] as isize) as i16;

    let mut k = 1us;
    while k < 64 {
        //let rs = try!(decode_huff(dc, &dc.ac_tables[comp.ac_table]));
        let rs = try!(decode_huff(dc, ac_table_idx, false));
        let rrrr = rs >> 4;
        let ssss = rs & 0xf;

        if ssss == 0 {
            if rrrr != 0xf {
                break;  // end of block
            }
            k += 16;    // run length is 16
            continue;
        }
        k += rrrr as usize;

        if 63 < k {
            return IFErr!("corrupt block");
        }
        res[DEZIGZAG[k] as usize] =
            (try!(receive_and_extend(dc, ssss)) * dc.qtables[qtable_idx][k] as isize) as i16;
        k += 1;
    }

    Ok(res)
}

//fn decode_huff<R: Reader>(dc: &mut JpegDecoder<R>, tab: &HuffTab) -> IoResult<u8> {
fn decode_huff<R: Reader>(dc: &mut JpegDecoder<R>, tab_idx: usize, dctab: bool) -> IoResult<u8> {
    let (code, cb, bits_left) = try!(nextbit(dc.stream, dc.cb, dc.bits_left));
    dc.cb = cb;
    dc.bits_left = bits_left;
    let mut code = code as i16;

    let mut i = 0;
    let tab: &HuffTab = if dctab { &dc.dc_tables[tab_idx] } else { &dc.ac_tables[tab_idx] };
    while tab.maxcode[i] < code {
        //code = (code << 1) + try!(nextbit(dc)) as i16;
        let (nb, cb, bits_left) = try!(nextbit(dc.stream, dc.cb, dc.bits_left));
        dc.cb = cb;
        dc.bits_left = bits_left;
        code = (code << 1) + nb as i16;

        i += 1;
        if tab.maxcode.len() <= i {
            return IFErr!("corrupt huffman coding");
        }
    }
    let j = (tab.valptr[i] + code - tab.mincode[i]) as usize;
    if tab.values.len() <= j {
        return IFErr!("corrupt huffman coding")
    }
    Ok(tab.values[j])
}

fn receive_and_extend<R: Reader>(dc: &mut JpegDecoder<R>, s: u8) -> IoResult<isize> {
    // receive
    let mut symbol = 0is;
    for _ in (0 .. s) {
        let (nb, cb, bits_left) = try!(nextbit(dc.stream, dc.cb, dc.bits_left));
        dc.cb = cb;
        dc.bits_left = bits_left;
        symbol = (symbol << 1) + nb as isize;

        //symbol = (symbol << 1) + try!(nextbit(dc)) as isize;
    }
    // extend
    let vt = 1 << (s as usize - 1);
    if symbol < vt {
        Ok(symbol + (-1 << s as usize) + 1)
    } else {
        Ok(symbol)
    }
}

// returns the bit and the new cb and bits_left
fn nextbit<R: Reader>(stream: &mut R, mut cb: u8, mut bits_left: usize)
                                           -> IoResult<(u8, u8, usize)>
{
    if bits_left == 0 {
        cb = try!(stream.read_u8());
        bits_left = 8;

        if cb == 0xff {
            let b2 = try!(stream.read_u8());
            if b2 != 0x0 {
                return IFErr!("unexpected marker")
            }
        }
    }

    let r = cb >> 7;
    cb <<= 1;
    bits_left -= 1;
    Ok((r, cb, bits_left))
}

fn reconstruct<R: Reader>(dc: &JpegDecoder<R>) -> IoResult<Vec<u8>> {
    let tgt_chans = dc.tgt_fmt.channels();
    let mut result: Vec<u8> = repeat(0).take(dc.w * dc.h * tgt_chans).collect();

    match (dc.num_comps, dc.tgt_fmt) {
        (3, ColFmt::RGB) | (3, ColFmt::RGBA) => {
            for ref comp in dc.comps.iter() {
                if comp.sfx != dc.hmax || comp.sfy != dc.vmax {
                    upsample_rgb(dc, &mut result[]);
                    return Ok(result);
                }
            }

            let mut si = 0us;
            let mut di = 0us;
            for _j in (0 .. dc.h) {
                for i in (0 .. dc.w) {
                    let pixel = ycbcr_to_rgb(
                        dc.comps[0].data[si+i],
                        dc.comps[1].data[si+i],
                        dc.comps[2].data[si+i],
                    );
                    copy_memory(&mut result[di..di+3], &pixel[]);
                    if dc.tgt_fmt == ColFmt::RGBA {
                        *result.get_mut(di+3).unwrap() = 255;
                    }
                    di += tgt_chans;
                }
                si += dc.num_mcu_x * dc.comps[0].sfx * 8;
            }
            return Ok(result);
        },
        (_, ColFmt::Y) | (_, ColFmt::YA) => {
            let comp = &dc.comps[0];
            if comp.sfx != dc.hmax || comp.sfy != dc.vmax {
                upsample_gray(dc, &mut result[]);
                return Ok(result);
            }

            // no resampling
            let mut si = 0us;
            let mut di = 0us;
            if dc.tgt_fmt == ColFmt::YA {
                for _j in (0 .. dc.h) {
                    for i in (0 .. dc.w) {
                        *result.get_mut(di  ).unwrap() = comp.data[si+i];
                        *result.get_mut(di+1).unwrap() = 255;
                        di += 2;
                    }
                    si += dc.num_mcu_x * comp.sfx * 8;
                }
            } else {    // FmtY
                for _j in (0 .. dc.h) {
                    copy_memory(&mut result[di..di+dc.w], &comp.data[si..si+dc.w]);
                    si += dc.num_mcu_x * comp.sfx * 8;
                    di += dc.w;
                }
            }
            return Ok(result);
        },
        (1, ColFmt::RGB) | (1, ColFmt::RGBA) => {
            let comp = &dc.comps[0];
            let mut si = 0us;
            let mut di = 0us;
            for _j in (0 .. dc.h) {
                for i in (0 .. dc.w) {
                    *result.get_mut(di  ).unwrap() = comp.data[si+i];
                    *result.get_mut(di+1).unwrap() = comp.data[si+i];
                    *result.get_mut(di+2).unwrap() = comp.data[si+i];
                    if dc.tgt_fmt == ColFmt::RGBA {
                        *result.get_mut(di+3).unwrap() = 255;
                    }
                    di += tgt_chans;
                }
                si += dc.num_mcu_x * comp.sfx * 8;
            }
            return Ok(result);
        },
        _ => return IFErr!("internal error"),
    }
}

fn upsample_gray<R: Reader>(dc: &JpegDecoder<R>, result: &mut[u8]) {
    let stride0 = dc.num_mcu_x * dc.comps[0].sfx * 8;
    let si0yratio = dc.comps[0].y as f64 / dc.h as f64;
    let si0xratio = dc.comps[0].x as f64 / dc.w as f64;
    let mut di = 0us;
    let tgt_chans = dc.tgt_fmt.channels();

    for j in (0 .. dc.h) {
        let si0 = (j as f64 * si0yratio).floor() as usize * stride0;
        for i in (0 .. dc.w) {
            result[di] = dc.comps[0].data[si0 + (i as f64 * si0xratio).floor() as usize];
            if dc.tgt_fmt == ColFmt::YA { result[di+1] = 255; }
            di += tgt_chans;
        }
    }
}

fn upsample_rgb<R: Reader>(dc: &JpegDecoder<R>, result: &mut[u8]) {
    let stride0 = dc.num_mcu_x * dc.comps[0].sfx * 8;
    let stride1 = dc.num_mcu_x * dc.comps[1].sfx * 8;
    let stride2 = dc.num_mcu_x * dc.comps[2].sfx * 8;
    let si0yratio = dc.comps[0].y as f64 / dc.h as f64;
    let si1yratio = dc.comps[1].y as f64 / dc.h as f64;
    let si2yratio = dc.comps[2].y as f64 / dc.h as f64;
    let si0xratio = dc.comps[0].x as f64 / dc.w as f64;
    let si1xratio = dc.comps[1].x as f64 / dc.w as f64;
    let si2xratio = dc.comps[2].x as f64 / dc.w as f64;

    let mut di = 0us;
    let tgt_chans = dc.tgt_fmt.channels();

    for j in (0 .. dc.h) {
        let si0 = (j as f64 * si0yratio).floor() as usize * stride0;
        let si1 = (j as f64 * si1yratio).floor() as usize * stride1;
        let si2 = (j as f64 * si2yratio).floor() as usize * stride2;

        for i in (0 .. dc.w) {
            let pixel = ycbcr_to_rgb(
                dc.comps[0].data[si0 + (i as f64 * si0xratio).floor() as usize],
                dc.comps[1].data[si1 + (i as f64 * si1xratio).floor() as usize],
                dc.comps[2].data[si2 + (i as f64 * si2xratio).floor() as usize],
            );
            copy_memory(&mut result[di..di+3], &pixel[]);
            if dc.tgt_fmt == ColFmt::RGBA { result[di+3] = 255; }
            di += tgt_chans;
        }
    }
}

fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> [u8; 3] {
    let cb = cb as f32;
    let cr = cr as f32;
    [clamp_to_u8(y as f32 + 1.402*(cr-128.0)),
     clamp_to_u8(y as f32 - 0.34414*(cb-128.0) - 0.71414*(cr-128.0)),
     clamp_to_u8(y as f32 + 1.772*(cb-128.0))]
}

fn clamp_to_u8(x: f32) -> u8 {
    if x < 0.0 { return 0; }
    if 255.0 < x { return 255; }
    x as u8
}

// ------------------------------------------------------------
// The IDCT stuff here (to the next dashed line) is copied and adapted from
// stb_image which is released under public domain.  Many thanks to stb_image
// author, Sean Barrett.
// Link: https://github.com/nothings/stb/blob/master/stb_image.h

// idct and level-shift
unsafe fn stbi_idct_block(mut dst: *mut u8, dst_stride: usize, data: &[i16]) {
    let d = data;
    let mut v: [i32; 64] = [0; 64];

    // columns
    for i in (0 .. 8us) {
        if d[i+ 8]==0 && d[i+16]==0 && d[i+24]==0 && d[i+32]==0 &&
           d[i+40]==0 && d[i+48]==0 && d[i+56]==0 {
            let dcterm = (d[i] as i32) << 2;
            v[i   ] = dcterm;
            v[i+ 8] = dcterm;
            v[i+16] = dcterm;
            v[i+24] = dcterm;
            v[i+32] = dcterm;
            v[i+40] = dcterm;
            v[i+48] = dcterm;
            v[i+56] = dcterm;
        } else {
            let mut t0 = 0; let mut t1 = 0;
            let mut t2 = 0; let mut t3 = 0;
            let mut x0 = 0; let mut x1 = 0;
            let mut x2 = 0; let mut x3 = 0;
            stbi_idct_1d(
                &mut t0, &mut t1, &mut t2, &mut t3,
                &mut x0, &mut x1, &mut x2, &mut x3,
                d[i+ 0] as i32, d[i+ 8] as i32, d[i+16] as i32, d[i+24] as i32,
                d[i+32] as i32, d[i+40] as i32, d[i+48] as i32, d[i+56] as i32
            );

            // constants scaled things up by 1<<12; let's bring them back
            // down, but keep 2 extra bits of precision
            x0 += 512; x1 += 512; x2 += 512; x3 += 512;
            v[i+ 0] = (x0+t3) >> 10;
            v[i+56] = (x0-t3) >> 10;
            v[i+ 8] = (x1+t2) >> 10;
            v[i+48] = (x1-t2) >> 10;
            v[i+16] = (x2+t1) >> 10;
            v[i+40] = (x2-t1) >> 10;
            v[i+24] = (x3+t0) >> 10;
            v[i+32] = (x3-t0) >> 10;
        }
    }

    for i in range_step(0, 64, 8) {
        let mut t0 = 0; let mut t1 = 0;
        let mut t2 = 0; let mut t3 = 0;
        let mut x0 = 0; let mut x1 = 0;
        let mut x2 = 0; let mut x3 = 0;
        stbi_idct_1d(
            &mut t0, &mut t1, &mut t2, &mut t3,
            &mut x0, &mut x1, &mut x2, &mut x3,
            v[i+0],v[i+1],v[i+2],v[i+3],v[i+4],v[i+5],v[i+6],v[i+7]
        );
        // constants scaled things up by 1<<12, plus we had 1<<2 from first
        // loop, plus horizontal and vertical each scale by sqrt(8) so together
        // we've got an extra 1<<3, so 1<<17 total we need to remove.
        // so we want to round that, which means adding 0.5 * 1<<17,
        // aka 65536. Also, we'll end up with -128 to 127 that we want
        // to encode as 0-255 by adding 128, so we'll add that before the shift
        x0 += 65536 + (128<<17);
        x1 += 65536 + (128<<17);
        x2 += 65536 + (128<<17);
        x3 += 65536 + (128<<17);

        *dst.offset(0) = stbi_clamp((x0+t3) >> 17);
        *dst.offset(7) = stbi_clamp((x0-t3) >> 17);
        *dst.offset(1) = stbi_clamp((x1+t2) >> 17);
        *dst.offset(6) = stbi_clamp((x1-t2) >> 17);
        *dst.offset(2) = stbi_clamp((x2+t1) >> 17);
        *dst.offset(5) = stbi_clamp((x2-t1) >> 17);
        *dst.offset(3) = stbi_clamp((x3+t0) >> 17);
        *dst.offset(4) = stbi_clamp((x3-t0) >> 17);

        dst = dst.offset(dst_stride as isize);
    }
}

fn stbi_clamp(x: i32) -> u8 {
   if x as u32 > 255 {
      if x < 0 { return 0; }
      if x > 255 { return 255; }
   }
   return x as u8;
}

fn stbi_idct_1d(t0: &mut i32, t1: &mut i32, t2: &mut i32, t3: &mut i32,
                 x0: &mut i32, x1: &mut i32, x2: &mut i32, x3: &mut i32,
        s0: i32, s1: i32, s2: i32, s3: i32, s4: i32, s5: i32, s6: i32, s7: i32)
{
   let mut p2 = s2;
   let mut p3 = s6;
   let mut p1 = (p2+p3) * f2f(0.5411961_f32);
   *t2 = p1 + p3 * f2f(-1.847759065_f32);
   *t3 = p1 + p2 * f2f( 0.765366865_f32);
   p2 = s0;
   p3 = s4;
   *t0 = fsh(p2+p3);
   *t1 = fsh(p2-p3);
   *x0 = *t0+*t3;
   *x3 = *t0-*t3;
   *x1 = *t1+*t2;
   *x2 = *t1-*t2;
   *t0 = s7;
   *t1 = s5;
   *t2 = s3;
   *t3 = s1;
   p3 = *t0+*t2;
   let mut p4 = *t1+*t3;
   p1 = *t0+*t3;
   p2 = *t1+*t2;
   let p5 = (p3+p4)*f2f( 1.175875602_f32);
   *t0 = *t0*f2f( 0.298631336_f32);
   *t1 = *t1*f2f( 2.053119869_f32);
   *t2 = *t2*f2f( 3.072711026_f32);
   *t3 = *t3*f2f( 1.501321110_f32);
   p1 = p5 + p1*f2f(-0.899976223_f32);
   p2 = p5 + p2*f2f(-2.562915447_f32);
   p3 = p3*f2f(-1.961570560_f32);
   p4 = p4*f2f(-0.390180644_f32);
   *t3 += p1+p4;
   *t2 += p2+p3;
   *t1 += p2+p4;
   *t0 += p1+p3;
}

#[inline(always)] fn f2f(x: f32) -> i32 { (x * 4096_f32 + 0.5) as i32 }
#[inline(always)] fn fsh(x: i32) -> i32 { x << 12 }

// ------------------------------------------------------------

type LineConverter = fn(&[u8], &mut[u8]);

fn get_converter(src_fmt: ColFmt, tgt_fmt: ColFmt) -> Option<LineConverter> {
    use self::ColFmt::*;
    match (src_fmt, tgt_fmt) {
        (ref s, ref t) if (*s == *t) => Some(copy_line),
        (Y, YA)      => Some(y_to_ya),
        (Y, RGB)     => Some(y_to_rgb),
        (Y, RGBA)    => Some(y_to_rgba),
        (Y, BGR)     => Some(Y_TO_BGR),
        (Y, BGRA)    => Some(Y_TO_BGRA),
        (YA, Y)      => Some(ya_to_y),
        (YA, RGB)    => Some(ya_to_rgb),
        (YA, RGBA)   => Some(ya_to_rgba),
        (YA, BGR)    => Some(YA_TO_BGR),
        (YA, BGRA)   => Some(YA_TO_BGRA),
        (RGB, Y)     => Some(rgb_to_y),
        (RGB, YA)    => Some(rgb_to_ya),
        (RGB, RGBA)  => Some(rgb_to_rgba),
        (RGB, BGR)   => Some(RGB_TO_BGR),
        (RGB, BGRA)  => Some(RGB_TO_BGRA),
        (RGBA, Y)    => Some(rgba_to_y),
        (RGBA, YA)   => Some(rgba_to_ya),
        (RGBA, RGB)  => Some(rgba_to_rgb),
        (RGBA, BGR)  => Some(RGBA_TO_BGR),
        (RGBA, BGRA) => Some(RGBA_TO_BGRA),
        (BGR, Y)     => Some(bgr_to_y),
        (BGR, YA)    => Some(bgr_to_ya),
        (BGR, RGB)   => Some(bgr_to_rgb),
        (BGR, RGBA)  => Some(bgr_to_rgba),
        (BGRA, Y)    => Some(bgra_to_y),
        (BGRA, YA)   => Some(bgra_to_ya),
        (BGRA, RGB)  => Some(bgra_to_rgb),
        (BGRA, RGBA) => Some(bgra_to_rgba),
        _ => None,
    }
}

fn copy_line(src_line: &[u8], tgt_line: &mut[u8]) {
    copy_memory(tgt_line, src_line);
}

fn luminance(r: u8, g: u8, b: u8) -> u8 {
    (0.21 * r as f32 + 0.64 * g as f32 + 0.15 * b as f32) as u8
}

fn y_to_ya(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0us;
    for s in (0us .. src_line.len()) {
        tgt_line[t  ] = src_line[s];
        tgt_line[t+1] = 255;
        t += 2;
    }
}

const Y_TO_BGR: LineConverter = y_to_rgb;
fn y_to_rgb(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0us;
    for s in (0us .. src_line.len()) {
        tgt_line[t  ] = src_line[s];
        tgt_line[t+1] = src_line[s];
        tgt_line[t+2] = src_line[s];
        t += 3;
    }
}

const Y_TO_BGRA: LineConverter = y_to_rgba;
fn y_to_rgba(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0us;
    for s in (0us .. src_line.len()) {
        tgt_line[t  ] = src_line[s];
        tgt_line[t+1] = src_line[s];
        tgt_line[t+2] = src_line[s];
        tgt_line[t+3] = 255;
        t += 4;
    }
}

fn ya_to_y(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0us;
    for s in range_step(0, src_line.len(), 2) {
        tgt_line[t] = src_line[s];
        t += 1;
    }
}

const YA_TO_BGR: LineConverter = ya_to_rgb;
fn ya_to_rgb(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0us;
    for s in range_step(0, src_line.len(), 2) {
        tgt_line[t  ] = src_line[s];
        tgt_line[t+1] = src_line[s];
        tgt_line[t+2] = src_line[s];
        t += 3;
    }
}

const YA_TO_BGRA: LineConverter = ya_to_rgba;
fn ya_to_rgba(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0us;
    for s in range_step(0, src_line.len(), 2) {
        tgt_line[t  ] = src_line[s];
        tgt_line[t+1] = src_line[s];
        tgt_line[t+2] = src_line[s];
        tgt_line[t+3] = 255;
        t += 4;
    }
}

fn rgb_to_y(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0us;
    for s in range_step(0, src_line.len(), 3) {
        tgt_line[t] = luminance(src_line[s], src_line[s+1], src_line[s+2]);
        t += 1;
    }
}

fn rgb_to_ya(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0us;
    for s in range_step(0, src_line.len(), 3) {
        tgt_line[t  ] = luminance(src_line[s], src_line[s+1], src_line[s+2]);
        tgt_line[t+1] = 255;
        t += 2;
    }
}

fn rgb_to_rgba(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0us;
    for s in range_step(0, src_line.len(), 3) {
        tgt_line[t  ] = src_line[s  ];
        tgt_line[t+1] = src_line[s+1];
        tgt_line[t+2] = src_line[s+2];
        tgt_line[t+3] = 255;
        t += 4;
    }
}

fn rgba_to_y(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0us;
    for s in range_step(0, src_line.len(), 4) {
        tgt_line[t] = luminance(src_line[s], src_line[s+1], src_line[s+2]);
        t += 1;
    }
}

fn rgba_to_ya(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0us;
    for s in range_step(0, src_line.len(), 4) {
        tgt_line[t  ] = luminance(src_line[s], src_line[s+1], src_line[s+2]);
        tgt_line[t+1] = src_line[s+3];
        t += 2;
    }
}

fn rgba_to_rgb(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0us;
    for s in range_step(0, src_line.len(), 4) {
        tgt_line[t  ] = src_line[s  ];
        tgt_line[t+1] = src_line[s+1];
        tgt_line[t+2] = src_line[s+2];
        t += 3;
    }
}

fn bgr_to_y(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0us;
    for s in range_step(0, src_line.len(), 3) {
        tgt_line[t  ] = luminance(src_line[s+2], src_line[s+1], src_line[s]);
        t += 1;
    }
}

fn bgr_to_ya(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0us;
    for s in range_step(0, src_line.len(), 3) {
        tgt_line[t  ] = luminance(src_line[s+2], src_line[s+1], src_line[s]);
        tgt_line[t+1] = 255;
        t += 2;
    }
}

const RGB_TO_BGR: LineConverter = bgr_to_rgb;
fn bgr_to_rgb(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0us;
    for s in range_step(0, src_line.len(), 3) {
        tgt_line[t  ] = src_line[s+2];
        tgt_line[t+1] = src_line[s+1];
        tgt_line[t+2] = src_line[s  ];
        t += 3;
    }
}

const RGB_TO_BGRA: LineConverter = bgr_to_rgba;
fn bgr_to_rgba(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0us;
    for s in range_step(0, src_line.len(), 3) {
        tgt_line[t  ] = src_line[s+2];
        tgt_line[t+1] = src_line[s+1];
        tgt_line[t+2] = src_line[s  ];
        tgt_line[t+3] = 255;
        t += 4;
    }
}

fn bgra_to_y(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0us;
    for s in range_step(0, src_line.len(), 4) {
        tgt_line[t] = luminance(src_line[s+2], src_line[s+1], src_line[s]);
        t += 1;
    }
}

fn bgra_to_ya(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0us;
    for s in range_step(0, src_line.len(), 4) {
        tgt_line[t  ] = luminance(src_line[s+2], src_line[s+1], src_line[s]);
        tgt_line[t+1] = src_line[s+3];
        t += 2;
    }
}

const RGBA_TO_BGR: LineConverter = bgra_to_rgb;
fn bgra_to_rgb(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0us;
    for s in range_step(0, src_line.len(), 4) {
        tgt_line[t  ] = src_line[s+2];
        tgt_line[t+1] = src_line[s+1];
        tgt_line[t+2] = src_line[s  ];
        t += 3;
    }
}

const RGBA_TO_BGRA: LineConverter = bgra_to_rgba;
fn bgra_to_rgba(src_line: &[u8], tgt_line: &mut[u8]) {
    let mut t = 0us;
    for s in range_step(0, src_line.len(), 4) {
        tgt_line[t  ] = src_line[s+2];
        tgt_line[t+1] = src_line[s+1];
        tgt_line[t+2] = src_line[s  ];
        tgt_line[t+3] = src_line[s+3];
        t += 4;
    }
}

// ------------------------------------------------------------

fn crc32be(data: &[u8]) -> [u8; 4] {
    Crc32::new().put(data).finish_be()
}

struct Crc32 {
    r: u32
}

impl Crc32 {
    fn new() -> Crc32 { Crc32 { r: 0xffff_ffff } }

    fn put<'a>(&'a mut self, bytes: &[u8]) -> &'a mut Crc32 {
        for &byte in bytes.iter() {
            let idx = byte ^ (self.r as u8);
            self.r = (self.r >> 8) ^ CRC32_TABLE[idx as usize];
        }
        self
    }

    fn finish_be(&mut self) -> [u8; 4] {
        let result = u32_to_be(self.r ^ 0xffff_ffff);
        self.r = 0xffff_ffff;
        result
    }
}

static CRC32_TABLE: [u32; 256] = [
    0x00000000, 0x77073096, 0xee0e612c, 0x990951ba,
    0x076dc419, 0x706af48f, 0xe963a535, 0x9e6495a3,
    0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988,
    0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91,
    0x1db71064, 0x6ab020f2, 0xf3b97148, 0x84be41de,
    0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
    0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec,
    0x14015c4f, 0x63066cd9, 0xfa0f3d63, 0x8d080df5,
    0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172,
    0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b,
    0x35b5a8fa, 0x42b2986c, 0xdbbbc9d6, 0xacbcf940,
    0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
    0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116,
    0x21b4f4b5, 0x56b3c423, 0xcfba9599, 0xb8bda50f,
    0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924,
    0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d,
    0x76dc4190, 0x01db7106, 0x98d220bc, 0xefd5102a,
    0x71b18589, 0x06b6b51f, 0x9fbfe4a5, 0xe8b8d433,
    0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818,
    0x7f6a0dbb, 0x086d3d2d, 0x91646c97, 0xe6635c01,
    0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e,
    0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457,
    0x65b0d9c6, 0x12b7e950, 0x8bbeb8ea, 0xfcb9887c,
    0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
    0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2,
    0x4adfa541, 0x3dd895d7, 0xa4d1c46d, 0xd3d6f4fb,
    0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0,
    0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9,
    0x5005713c, 0x270241aa, 0xbe0b1010, 0xc90c2086,
    0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
    0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4,
    0x59b33d17, 0x2eb40d81, 0xb7bd5c3b, 0xc0ba6cad,
    0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1d29a,
    0xead54739, 0x9dd277af, 0x04db2615, 0x73dc1683,
    0xe3630b12, 0x94643b84, 0x0d6d6a3e, 0x7a6a5aa8,
    0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1,
    0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe,
    0xf762575d, 0x806567cb, 0x196c3671, 0x6e6b06e7,
    0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc,
    0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5,
    0xd6d6a3e8, 0xa1d1937e, 0x38d8c2c4, 0x4fdff252,
    0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
    0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60,
    0xdf60efc3, 0xa867df55, 0x316e8eef, 0x4669be79,
    0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236,
    0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f,
    0xc5ba3bbe, 0xb2bd0b28, 0x2bb45a92, 0x5cb36a04,
    0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
    0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a,
    0x9c0906a9, 0xeb0e363f, 0x72076785, 0x05005713,
    0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0x0cb61b38,
    0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21,
    0x86d3d2d4, 0xf1d4e242, 0x68ddb3f8, 0x1fda836e,
    0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
    0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c,
    0x8f659eff, 0xf862ae69, 0x616bffd3, 0x166ccf45,
    0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2,
    0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db,
    0xaed16a4a, 0xd9d65adc, 0x40df0b66, 0x37d83bf0,
    0xa9bcae53, 0xdebb9ec5, 0x47b2cf7f, 0x30b5ffe9,
    0xbdbdf21c, 0xcabac28a, 0x53b39330, 0x24b4a3a6,
    0xbad03605, 0xcdd70693, 0x54de5729, 0x23d967bf,
    0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94,
    0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d
];

// ------------------------------------------------------------

fn u16_from_be(buf: &[u8]) -> u16 {
    (buf[0] as u16) << 8 | buf[1] as u16
}

fn u16_from_le(buf: &[u8]) -> u16 {
    (buf[1] as u16) << 8 | buf[0] as u16
}

fn u16_to_le(x: u16) -> [u8; 2] {
    let buf = [x as u8, (x >> 8) as u8];
    buf
}

fn u32_from_be(buf: &[u8]) -> u32 {
    (buf[0] as u32) << 24 | (buf[1] as u32) << 16 | (buf[2] as u32) << 8 | buf[3] as u32
}

fn u32_to_be(x: u32) -> [u8; 4] {
    let buf = [(x >> 24) as u8, (x >> 16) as u8,
               (x >>  8) as u8, (x)       as u8];
    buf
}

fn skip<R: Reader>(stream: &mut R, mut bytes: usize) -> IoResult<()> {
    let mut buf = [0u8; 1024];
    while 0 < bytes {
        let n = min(bytes, buf.len());
        try!(stream.read_at_least(n, &mut buf[0..n]));
        bytes -= n;
    }
    Ok(())
}

fn extract_extension(filename: &str) -> Option<&str> {
    match filename.rfind('.') {
        Some(i) => Some(&filename[i..]),
        None => None,
    }
}
