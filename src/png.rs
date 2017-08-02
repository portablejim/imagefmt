// Copyright (c) 2014-2015 Tero Hänninen, license: MIT

extern crate deflate;
extern crate inflate;
use std::io::{Read, Write, Seek, SeekFrom};
use std::iter::{repeat};
use std::mem::{self, size_of};
use {
    Image, Info, ColFmt, ColType,
    converter, u32_to_be, u32_from_be,
};
use internal::Sample;
use self::internal::{ToVec, Buffer};

mod internal {
    use ::internal::Sample;

    pub enum Buffer {
        Bpc8(Vec<u8>),
        Bpc16(Vec<u16>),
    }

    pub trait ToVec<T: Sample> {
        fn vec(self) -> ::Result<Vec<T>>;
    }

    impl ToVec<u8> for Buffer {
        fn vec(self) -> ::Result<Vec<u8>> {
            match self {
                Buffer::Bpc8(b) => Ok(b),
                _ => return Err(::Error::Internal("bug")),
            }
        }
    }

    impl ToVec<u16> for Buffer {
        fn vec(self) -> ::Result<Vec<u16>> {
            match self {
                Buffer::Bpc16(b) => Ok(b),
                _ => return Err(::Error::Internal("bug")),
            }
        }
    }
}

/// Header of a PNG image.
#[derive(Debug)]
struct PngHeader {
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
pub fn read_info<R: Read>(reader: &mut R) -> ::Result<Info> {
    let hdr = try!(read_header(reader));

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
fn read_header<R: Read>(reader: &mut R) -> ::Result<PngHeader> {
    let mut buf = [0u8; 33];  // file header + IHDR
    try!(reader.read_exact(&mut buf));

    if &buf[0..8] != &PNG_FILE_HEADER[..] ||
       &buf[8..16] != b"\0\0\0\x0dIHDR" ||
       &buf[29..33] != &Crc32::new().put(&buf[12..29]).finish_be()[..]
    {
        return Err(::Error::InvalidData("corrupt png header"))
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

pub fn detect<R: Read+Seek>(reader: &mut R) -> bool {
    let mut buf = [0u8; 8];
    let start = match reader.seek(SeekFrom::Current(0))
        { Ok(s) => s, Err(_) => return false };
    let result = reader.read_exact(&mut buf).is_ok()
              && &buf[0..8] == &PNG_FILE_HEADER[..];
    let _ = reader.seek(SeekFrom::Start(start));
    result
}

/// Reads an image and converts it to requested format.
///
/// Passing `ColFmt::Auto` as `req_fmt` converts the data to one of `Y`, `YA`, `RGB`,
/// `RGBA`.  Paletted images are auto-depaletted.
#[inline]
pub fn read<R, T>(reader: &mut R, req_fmt: ColFmt) -> ::Result<Image<T>>
        where R: Read+Seek, T: Sample, Buffer: ToVec<T>
{
    Ok(try!(read_chunks(reader, req_fmt, &[])).0)
}

/// Like `png::read` but also returns the requested extension chunks.
///
/// If the requested chunks are not present they are ignored.
pub fn read_chunks<R, T>(reader: &mut R, req_fmt: ColFmt, chunk_names: &[[u8; 4]])
                                            -> ::Result<(Image<T>, Vec<ExtChunk>)>
        where R: Read+Seek, T: Sample, Buffer: ToVec<T>
{
    let dc = &mut try!(init_decoder(reader, req_fmt, size_of::<T>() * 8));
    let (buf, chunks) = try!(decode(dc, chunk_names));
    Ok((Image {
       w  : dc.w,
       h  : dc.h,
       fmt: dc.tgt_fmt,
       buf: try!(buf.vec()),
    }, chunks))
}

fn init_decoder<R: Read+Seek>(reader: &mut R, req_fmt: ColFmt, req_bpc: usize)
                                                  -> ::Result<PngDecoder<R>>
{
    let hdr = try!(read_header(reader));

    if hdr.width < 1 || hdr.height < 1 {
        return Err(::Error::InvalidData("invalid dimensions"))
    }
    if (hdr.bit_depth != 8 && hdr.bit_depth != 16) || (req_bpc != 8 && req_bpc != 16) {
        return Err(::Error::Unsupported("only 8-bit and 16-bit images supported"))
    }
    if hdr.compression_method != 0 || hdr.filter_method != 0 {
        return Err(::Error::Unsupported("compression method"))
    }

    let ilace = match PngInterlace::from_u8(hdr.interlace_method) {
        Some(im) => im,
        None => return Err(::Error::Unsupported("interlace method")),
    };

    let src_fmt = match hdr.color_type {
        0 => ColFmt::Y,
        2 => ColFmt::RGB,
        3 => ColFmt::RGB,   // format of the palette
        4 => ColFmt::YA,
        6 => ColFmt::RGBA,
        _ => return Err(::Error::Unsupported("color type")),
    };

    Ok(PngDecoder {
        stream      : reader,
        w           : hdr.width as usize,
        h           : hdr.height as usize,
        ilace       : ilace,
        src_bpc     : hdr.bit_depth as usize,
        tgt_bpc     : req_bpc,
        src_indexed : hdr.color_type == PngColortype::Idx as u8,
        src_fmt     : src_fmt,
        tgt_fmt     : if req_fmt == ColFmt::Auto { src_fmt } else { req_fmt },
        chunk_lentype : [0u8; 8],
        crc         : Crc32::new(),
    })
}

struct PngDecoder<'r, R:'r> {
    stream        : &'r mut R,
    w             : usize,
    h             : usize,
    ilace         : PngInterlace,
    src_bpc       : usize,    // bits per channel
    tgt_bpc       : usize,
    src_indexed   : bool,
    src_fmt       : ColFmt,
    tgt_fmt       : ColFmt,

    chunk_lentype: [u8; 8],   // for reading len, type
    crc: Crc32,
}

#[derive(Eq, PartialEq)]
enum PngStage {
    IhdrParsed,
    PlteParsed,
    IdatParsed,
    //IendParsed,
}

fn read_chunkmeta<R: Read>(dc: &mut PngDecoder<R>) -> ::Result<usize> {
    try!(dc.stream.read_exact(&mut dc.chunk_lentype[0..8]));
    let len = u32_from_be(&dc.chunk_lentype[0..4]) as usize;
    if 0x7fff_ffff < len { return Err(::Error::InvalidData("chunk too long")) }
    dc.crc.put(&dc.chunk_lentype[4..8]);   // type
    Ok(len)
}

#[inline]
fn readcheck_crc<R: Read>(dc: &mut PngDecoder<R>) -> ::Result<()> {
    let mut tmp = [0u8; 4];
    try!(dc.stream.read_exact(&mut tmp));
    if &dc.crc.finish_be()[..] != &tmp[0..4] {
        return Err(::Error::InvalidData("corrupt chunk"))
    }
    Ok(())
}

fn decode<R: Read+Seek>(dc: &mut PngDecoder<R>, chunk_names: &[[u8; 4]])
                                  -> ::Result<(Buffer, Vec<ExtChunk>)>
{
    use self::PngStage::*;

    let mut result = Buffer::Bpc8(Vec::<u8>::new());
    let mut chunks = Vec::<ExtChunk>::new();
    let mut palette = Vec::<u8>::new();

    let mut stage = IhdrParsed;

    let mut len = try!(read_chunkmeta(dc));

    loop {
        match &dc.chunk_lentype[4..8] {
            b"IDAT" => {
                if !(stage == IhdrParsed || (stage == PlteParsed && dc.src_indexed)) {
                    return Err(::Error::InvalidData("corrupt chunk stream"))
                }

                // also reads chunk_lentype for next chunk
                result = try!(read_idat_stream(dc, &mut len, &palette[..]));
                stage = IdatParsed;
                continue;   // skip reading chunk_lentype
            }
            b"PLTE" => {
                let entries = len / 3;
                if stage != IhdrParsed || len % 3 != 0 || 256 < entries {
                    return Err(::Error::InvalidData("corrupt chunk stream"))
                }
                palette = vec![0u8; len];
                try!(dc.stream.read_exact(&mut palette));
                dc.crc.put(&palette[..]);
                try!(readcheck_crc(dc));
                stage = PlteParsed;
            }
            b"IEND" => {
                if stage != IdatParsed {
                    return Err(::Error::InvalidData("corrupt chunk stream"))
                }
                let mut crc = [0u8; 4];
                try!(dc.stream.read_exact(&mut crc));
                if len != 0 || &crc[0..4] != &[0xae, 0x42, 0x60, 0x82][..] {
                    return Err(::Error::InvalidData("corrupt chunk"))
                }
                break;//stage = IendParsed;
            }
            _ => {
                if chunk_names.iter().any(|name| &name[..] == &dc.chunk_lentype[4..8]) {
                    let name = [dc.chunk_lentype[4], dc.chunk_lentype[5],
                                dc.chunk_lentype[6], dc.chunk_lentype[7]];
                    let mut data = vec![0u8; len];
                    try!(dc.stream.read_exact(&mut data));
                    dc.crc.put(&data[..]);
                    chunks.push(ExtChunk { name: name, data: data });
                    try!(readcheck_crc(dc));
                } else {
                    // unknown chunk, skip data and crc, and reset crc
                    try!(dc.stream.seek(SeekFrom::Current(len as i64 + 4)));
                    dc.crc.finish_be();
                }
            }
        }

        len = try!(read_chunkmeta(dc));
    }

    Ok((result, chunks))
}

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
        dst[d..d+3].copy_from_slice(&palette[pidx..pidx+3]);
        d += 3;
    }
}

fn read_idat_stream<R: Read>(dc: &mut PngDecoder<R>, len: &mut usize, palette: &[u8])
                                                                  -> ::Result<Buffer>
{
    let filter_step = match (dc.src_indexed, dc.src_bpc) {
        (true, _) => 1,
        (false, 8) => dc.src_fmt.channels(),
        (false, 16) => dc.src_fmt.channels() * 2,
        _ => return Err(::Error::Internal("unsupported bit depth"))
    };
    let max_src_linesz = dc.w * filter_step;
    let src_chans = dc.src_fmt.channels();
    let tgt_chans = dc.tgt_fmt.channels();
    let srclen = dc.w * src_chans;
    let tgtlen = dc.w * tgt_chans;

    let mut cline = vec![0u8; max_src_linesz+1];   // current line, incl. filter type byte
    let mut pline = vec![0u8; max_src_linesz+1];   // previous line
    let mut depaletted = if dc.src_indexed { vec![0u8; dc.w * 3] } else { Vec::new() };
    let mut cline8  = if dc.tgt_bpc == 8 { vec![0u8; srclen] } else { Vec::new() };
    let mut cline16 = if dc.tgt_bpc == 16 { vec![0u16; srclen] } else { Vec::new() };
    let mut result8 = if dc.tgt_bpc == 8 { vec![0u8; dc.h * tgtlen] } else { Vec::new() };
    let mut result16 = if dc.tgt_bpc == 16 { vec![0u16; dc.h*tgtlen] } else { Vec::new()};

    let (convert8, _0, _1, _2, _3) = try!(converter::<u8>(dc.src_fmt, dc.tgt_fmt));
    let (convert16, c0, c1, c2, c3) = try!(converter::<u16>(dc.src_fmt, dc.tgt_fmt));

    let compressed_data = try!(read_idat_chunks(dc, len));
    let mut zlib = &*inflate::inflate_bytes_zlib(&compressed_data[..]).expect("Error decoding stream");

    match dc.ilace {
        PngInterlace::None => {
            let mut ti = 0;
            for _j in 0 .. dc.h {
                try!(zlib.read_exact(&mut cline[..]));
                let filter_type: u8 = cline[0];

                try!(recon(&mut cline[1..], &pline[1..], filter_type, filter_step));

                {
                    let bytes: &[u8] =
                        if dc.src_indexed {
                            depalettize(&cline[1..], &palette, &mut depaletted);
                            &depaletted[..]
                        } else {
                            &cline[1..]
                        };

                    if dc.tgt_bpc == 8 {
                        bpc8_from_bytes(&bytes, dc.src_bpc, &mut cline8);
                        convert8(&cline8, &mut result8[ti..ti+tgtlen], c0, c1, c2, c3);
                    } else {
                        bpc16_from_bytes(&bytes, dc.src_bpc, &mut cline16);
                        convert16(&cline16, &mut result16[ti..ti+tgtlen], c0, c1, c2, c3);
                    }
                }

                ti += tgtlen;

                mem::swap(&mut cline, &mut pline);
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

            let mut redline8  = if dc.tgt_bpc == 8 { vec![0u8; tgtlen] } else { Vec::new() };
            let mut redline16 = if dc.tgt_bpc == 16 { vec![0u16; tgtlen] } else { Vec::new() };

            for pass in 0..7 {
                let tgt_px    = A7_IDX_TRANSLATORS[pass];   // target pixel
                let redlinesz = redw[pass] * filter_step;
                let mut cline = &mut cline[..redlinesz+1];
                let mut pline = &mut pline[..redlinesz+1];
                for b in &mut pline[..] { *b = 0 }  // For defiltering.

                for j in 0 .. redh[pass] {
                    try!(zlib.read_exact(&mut cline[..]));
                    let filter_type: u8 = cline[0];

                    try!(recon(&mut cline[1..], &pline[1..], filter_type, filter_step));

                    {
                        let bytes: &[u8] =
                            if dc.src_indexed {
                                depalettize(&cline[1..], &palette, &mut depaletted);
                                &depaletted[..]
                            } else {
                                &cline[1..]
                            };

                        let mut redi = 0;
                        if dc.tgt_bpc == 8 {
                            bpc8_from_bytes(&bytes, dc.src_bpc, &mut cline8);
                            convert8(&cline8[..redw[pass] * src_chans],
                                     &mut redline8[0..redw[pass] * tgt_chans],
                                     c0, c1, c2, c3);
                            for i in 0 .. redw[pass] {
                                let tgt = tgt_px(i, j, dc.w) * tgt_chans;
                                result8[tgt .. tgt+tgt_chans]
                                    .copy_from_slice(&redline8[redi .. redi+tgt_chans]);
                                redi += tgt_chans;
                            }
                        } else {
                            bpc16_from_bytes(&bytes, dc.src_bpc, &mut cline16);
                            convert16(&cline16[..redw[pass] * src_chans],
                                      &mut redline16[0..redw[pass] * tgt_chans],
                                      c0, c1, c2, c3);
                            for i in 0 .. redw[pass] {
                                let tgt = tgt_px(i, j, dc.w) * tgt_chans;
                                result16[tgt .. tgt+tgt_chans]
                                    .copy_from_slice(&redline16[redi .. redi+tgt_chans]);
                                redi += tgt_chans;
                            }
                        }
                    }

                    mem::swap(&mut cline, &mut pline);
                }
            }
        } // Adam7
    }

    Ok(if dc.tgt_bpc == 8 { Buffer::Bpc8(result8) } else { Buffer::Bpc16(result16) })
}

fn bpc8_from_bytes(src: &[u8], bpc: usize, tgt: &mut[u8]) {
    match bpc {
        8 => tgt[..src.len()].copy_from_slice(&src),  // This copy is unnecessary, but
                                                      // it's hard to find nice way to
                                                      // avoid it.
        16 => {
            let mut s = 0;
            let mut t = 0;
            while s < src.len() {
                tgt[t] = src[s];    // truncate
                s += 2;
                t += 1;
            }
        }
        _ => panic!("bug"),
    }
}

fn bpc16_from_bytes(src: &[u8], bpc: usize, tgt: &mut[u16]) {
    match bpc {
        8 => {
            for k in 0..src.len() {
                tgt[k] = src[k] as u16 * 256 + 128;
            }
        }
        16 => {
            let mut s = 0;
            let mut t = 0;
            while s < src.len() {
                tgt[t] = (src[s] as u16) << 8 | src[s+1] as u16;
                s += 2;
                t += 1;
            }
        }
        _ => panic!("bug"),
    }
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
fn read_idat_chunks<R: Read>(dc: &mut PngDecoder<R>, len: &mut usize)
                                               -> ::Result<(Vec<u8>)>
{
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

fn recon(cline: &mut[u8], pline: &[u8], ftype: u8, fstep: usize) -> ::Result<()> {
    match PngFilter::from_u8(ftype) {
        Some(PngFilter::None)
            => { }
        Some(PngFilter::Sub) => {
            unsafe {
                for k in fstep .. cline.len() {
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
                for k in fstep .. cline.len() {
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
                for k in fstep .. cline.len() {
                    *cline.get_unchecked_mut(k) =
                        (*cline.get_unchecked(k)).wrapping_add(
                            paeth(*cline.get_unchecked(k-fstep),
                                        *pline.get_unchecked(k),
                                          *pline.get_unchecked(k-fstep)));
                }
            }
        }
        None => return Err(::Error::InvalidData("invalid filter type")),
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
pub struct ExtChunk {
    pub name: [u8; 4],
    pub data: Vec<u8>,
}

/// Writes an image and converts it to requested color type.
#[inline]
pub fn write<W: Write>(writer: &mut W, w: usize, h: usize, src_fmt: ColFmt, data: &[u8],
                                                                      tgt_type: ColType,
                                                              src_stride: Option<usize>)
                                                                         -> ::Result<()>
{
    write_chunks(writer, w, h, src_fmt, data, tgt_type, src_stride, &[])
}

/// Like `png::write` but also writes the given extension chunks.
pub fn write_chunks<W: Write>(writer: &mut W, w: usize, h: usize, src_fmt: ColFmt,
                                                                      data: &[u8],
                                                                tgt_type: ColType,
                                                        src_stride: Option<usize>,
                                                              chunks: &[ExtChunk])
                                                                   -> ::Result<()>
{
    if src_fmt == ColFmt::Auto { return Err(::Error::InvalidArg("invalid format")) }
    let stride = src_stride.unwrap_or(w * src_fmt.channels());

    if w < 1 || h < 1
    || (src_stride.is_none() && src_fmt.channels() * w * h != data.len())
    || (src_stride.is_some() && data.len() < stride * (h-1) + w * src_fmt.channels()) {
        return Err(::Error::InvalidArg("invalid dimensions or data length"))
    }

    let tgt_fmt = match tgt_type {
        ColType::Gray       => ColFmt::Y,
        ColType::GrayAlpha  => ColFmt::YA,
        ColType::Color      => ColFmt::RGB,
        ColType::ColorAlpha => ColFmt::RGBA,
        ColType::Auto => match src_fmt {
            ColFmt::Y                 => ColFmt::Y,
            ColFmt::YA   | ColFmt::AY => ColFmt::YA,
            ColFmt::RGB  | ColFmt::BGR => ColFmt::RGB,
            ColFmt::RGBA | ColFmt::BGRA => ColFmt::RGBA,
            ColFmt::ARGB | ColFmt::ABGR => ColFmt::RGBA,
            ColFmt::Auto => return Err(::Error::InvalidArg("invalid format")),
        },
    };

    let ec = &mut PngEncoder {
        stream    : writer,
        w         : w,
        h         : h,
        src_stride: stride,
        src_fmt   : src_fmt,
        tgt_fmt   : tgt_fmt,
        data      : data,
        crc       : Crc32::new(),
    };

    try!(write_header(ec));
    for chunk in chunks {
        try!(write_custom_chunk(ec, chunk));
    }
    try!(write_image_data(ec));

    let iend: &'static[u8] = b"\0\0\0\0IEND\xae\x42\x60\x82";
    try!(ec.stream.write_all(iend));
    try!(ec.stream.flush());
    Ok(())
}

fn write_header<W: Write>(ec: &mut PngEncoder<W>) -> ::Result<()> {
    let mut crc = Crc32::new();
    let width = &u32_to_be(ec.w as u32)[..];
    let height = &u32_to_be(ec.h as u32)[..];

    try!(ec.stream.write_all(&PNG_FILE_HEADER[..]));
    try!(ec.stream.write_all(b"\0\0\0\x0dIHDR"));     crc.put(b"IHDR");
    try!(ec.stream.write_all(width));                 crc.put(width);
    try!(ec.stream.write_all(height));                crc.put(height);
    let tmp = [
        8,          // bit depth
        match ec.tgt_fmt {    // color type
            ColFmt::Y => PngColortype::Y,
            ColFmt::YA => PngColortype::YA,
            ColFmt::RGB => PngColortype::RGB,
            ColFmt::RGBA => PngColortype::RGBA,
            _ => return Err(::Error::Internal("wrong format")),
        } as u8,
        0, 0, 0     // compression, filter, interlace
    ];
    try!(ec.stream.write_all(&tmp));
    crc.put(&tmp);

    try!(ec.stream.write_all(&crc.finish_be()[..]));
    Ok(())
}

fn write_custom_chunk<W: Write>(ec: &mut PngEncoder<W>, chunk: &ExtChunk)
                                                          -> ::Result<()>
{
    if chunk.name[0] < b'a' || b'z' < chunk.name[0] {
        return Err(::Error::InvalidArg("invalid chunk name"))
    }
    for &b in &chunk.name[1..] {
        if b < b'A' || (b'Z' < b && b < b'a') || b'z' < b {
            return Err(::Error::InvalidArg("invalid chunk name"));
        }
    }
    if 0x7fff_ffff < chunk.data.len() {
        return Err(::Error::InvalidData("chunk too long"))
    }

    try!(ec.stream.write_all(&u32_to_be(chunk.data.len() as u32)[..]));
    try!(ec.stream.write_all(&chunk.name[..]));
    try!(ec.stream.write_all(&chunk.data[..]));
    let mut crc = Crc32::new();
    crc.put(&chunk.name[..]);
    crc.put(&chunk.data[..]);
    try!(ec.stream.write_all(&crc.finish_be()[..]));
    Ok(())
}

struct PngEncoder<'r, W:'r> {
    stream        : &'r mut W,
    w             : usize,
    h             : usize,
    src_stride    : usize,
    tgt_fmt       : ColFmt,
    src_fmt       : ColFmt,
    data          : &'r [u8],
    crc           : Crc32,
}

fn write_image_data<W: Write>(ec: &mut PngEncoder<W>) -> ::Result<()> {
    let (convert, c0, c1, c2, c3) = try!(converter(ec.src_fmt, ec.tgt_fmt));

    let fstep = ec.tgt_fmt.channels();   // filter step
    let tgt_linesz = ec.w * fstep + 1;   // +1 for filter type
    let mut cline = vec![0u8; tgt_linesz];
    let mut pline = vec![0u8; tgt_linesz];
    let mut filtered_image = vec![0u8; tgt_linesz * ec.h];

    let src_linesz = ec.w * ec.src_fmt.channels();

    let mut si = 0;
    let mut ti = 0;
    while si < ec.h * ec.src_stride {
        convert(&ec.data[si .. si+src_linesz], &mut cline[1 .. tgt_linesz],
                c0, c1, c2, c3);

        for i in 1 .. fstep+1 {
            filtered_image[ti+i] = cline[i].wrapping_sub(paeth(0, pline[i], 0));
        }
        for i in fstep+1 .. cline.len() {
            filtered_image[ti+i] = cline[i]
                .wrapping_sub(paeth(cline[i-fstep], pline[i], pline[i-fstep]));
        }

        filtered_image[ti] = PngFilter::Paeth as u8;

        mem::swap(&mut cline, &mut pline);

        si += ec.src_stride;
        ti += tgt_linesz;
    }

    let mut zlibenc = deflate::write::ZlibEncoder::new(Vec::new(), deflate::Compression::Best);
    zlibenc.write_all(&filtered_image).expect("Write error");
    let compressed = zlibenc.finish().expect("Failed to finish the encoding");

    for idat_chunk in compressed.chunks(1024*32) {
        ec.crc.put(b"IDAT");
        ec.crc.put(&compressed);
        let crc = &ec.crc.finish_be();
        try!(ec.stream.write_all(&u32_to_be(idat_chunk.len() as u32)));
        try!(ec.stream.write_all(b"IDAT"));
        try!(ec.stream.write_all(&compressed));
        try!(ec.stream.write_all(crc));
    }

    Ok(())
}

struct Crc32 {
    r: u32
}

impl Crc32 {
    fn new() -> Crc32 { Crc32 { r: 0xffff_ffff } }

    fn put<'a>(&'a mut self, bytes: &[u8]) -> &'a mut Crc32 {
        for &byte in bytes {
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
