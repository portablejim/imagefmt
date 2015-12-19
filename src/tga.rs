// Copyright (c) 2014-2015 Tero Hänninen, license: MIT

use std::io::{Read, Write, Seek, SeekFrom};
use std::cmp::min;
use super::{
    Image, Info, ColFmt, ColType,
    copy_memory, converter,
    u16_to_le, u16_from_le, IFRead,
};

/// Header of a TGA image.
#[derive(Debug)]
struct TgaHeader {
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

/// Returns width, height and color type of the image.
pub fn read_info<R: Read>(reader: &mut R) -> ::Result<Info> {
    let hdr = try!(read_header(reader));
    let TgaInfo { src_fmt, .. } = try!(parse_header(&hdr));

    Ok(Info {
        w: hdr.width as usize,
        h: hdr.height as usize,
        ct: src_fmt.color_type(),
    })
}

/// Reads a TGA header.
///
/// The fields are not parsed into enums or anything like that.
fn read_header<R: Read>(reader: &mut R) -> ::Result<TgaHeader> {
    let mut buf = [0u8; 18];
    try!(reader.read_exact_(&mut buf));

    let hdr = TgaHeader {
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
    };

    if hdr.width < 1 && hdr.height < 1 && hdr.palette_type > 1
    || (hdr.palette_type == 0 && (hdr.palette_start > 0 ||
                                  hdr.palette_length > 0 ||
                                  hdr.palette_bits > 0))
    || !match hdr.data_type { 1...3 | 9...11 => true, _ => false } {
        Err(::Error::InvalidData("corrupt TGA header"))
    } else {
        Ok(hdr)
    }
}

pub fn detect<R: Read+Seek>(reader: &mut R) -> bool {
    let start = match reader.seek(SeekFrom::Current(0))
        { Ok(s) => s, Err(_) => return false };
    let result = read_header(reader).is_ok();
    let _ = reader.seek(SeekFrom::Start(start));
    result
}

/// Reads an image and converts it to requested format.
///
/// Passing `ColFmt::Auto` as req_fmt converts the data to one of `Y`, `YA`, `RGB`,
/// `RGBA`.
pub fn read<R: Read+Seek>(reader: &mut R, req_fmt: ColFmt) -> ::Result<Image> {
    let hdr = try!(read_header(reader));

    if 0 < hdr.palette_type { return Err(::Error::Unsupported("paletted TGAs not supported")) }
    if hdr.width < 1 || hdr.height < 1 { return Err(::Error::InvalidData("invalid dimensions")) }
    if 0 < (hdr.flags & 0xc0) { return Err(::Error::Unsupported("interlaced TGAs not supported")) }
    if 0 < (hdr.flags & 0x10) { return Err(::Error::Unsupported("right-to-left TGAs not supported")) }

    let TgaInfo { src_fmt, rle } = try!(parse_header(&hdr));

    let tgt_fmt = {
        use super::ColFmt::*;
        match req_fmt {
            Auto => match src_fmt {
                Y => Y,
                YA => YA,
                BGR => RGB,
                BGRA => RGBA,
                _ => return Err(::Error::Internal("wrong format")),
            },
            _ => req_fmt,
        }
    };

    try!(reader.seek(SeekFrom::Current(hdr.id_length as i64)));

    let dc = &mut TgaDecoder {
        stream         : reader,
        w              : hdr.width as usize,
        h              : hdr.height as usize,
        origin_at_top  : 0 < (hdr.flags & 0x20),
        rle            : rle,
        src_fmt        : src_fmt,
        tgt_fmt        : tgt_fmt,
    };

    Ok(Image {
        w   : dc.w,
        h   : dc.h,
        fmt : dc.tgt_fmt,
        buf : try!(decode(dc))
    })
}

struct TgaInfo {
    src_fmt   : ColFmt,
    rle       : bool,
}

fn parse_header(hdr: &TgaHeader) -> ::Result<TgaInfo> {
    use self::TgaDataType::*;

    let mut rle = false;
    let datatype = TgaDataType::from_u8(hdr.data_type);
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
        _ => return Err(::Error::Unsupported("data type"))
    }

    let src_fmt = match hdr.bits_pp as usize / 8 {
        1 => ColFmt::Y,
        2 => ColFmt::YA,
        3 => ColFmt::BGR,
        4 => ColFmt::BGRA,
        _ => return Err(::Error::Unsupported("format")),
    };

    Ok(TgaInfo {
        src_fmt   : src_fmt,
        rle       : rle,
    })
}

fn decode<R: Read>(dc: &mut TgaDecoder<R>) -> ::Result<Vec<u8>> {
    let tgt_linesz = (dc.w * dc.tgt_fmt.channels()) as isize;
    let src_linesz = dc.w * dc.src_fmt.channels();

    let mut src_line = vec![0u8; src_linesz];
    let mut result = vec![0u8; dc.h * dc.w * dc.tgt_fmt.channels()];

    let (tgt_stride, mut ti): (isize, isize) =
        if dc.origin_at_top {
            (tgt_linesz, 0)
        } else {
            (-tgt_linesz, (dc.h-1) as isize * tgt_linesz)
        };

    let (convert, c0, c1, c2, c3) = try!(converter(dc.src_fmt, dc.tgt_fmt));

    if !dc.rle {
        for _j in (0 .. dc.h) {
            try!(dc.stream.read_exact_(&mut src_line[0..src_linesz]));
            convert(&src_line[..], &mut result[ti as usize..(ti+tgt_linesz) as usize],
                    c0, c1, c2, c3);
            ti += tgt_stride;
        }
        return Ok(result);
    }

    // ---- RLE ----

    let bytes_pp = dc.src_fmt.channels();
    let mut rbuf = vec![0u8; src_linesz];
    let mut plen = 0;    // packet length
    let mut its_rle = false;

    for _ in (0 .. dc.h) {
        // fill src_line with uncompressed data
        let mut wanted: usize = src_linesz;
        while 0 < wanted {
            if plen == 0 {
                let hdr = try!(dc.stream.read_u8()) as usize;
                its_rle = 0 < (hdr & 0x80);
                plen = ((hdr & 0x7f) + 1) * bytes_pp;
            }
            let gotten: usize = src_linesz - wanted;
            let copysize: usize = min(plen, wanted);
            if its_rle {
                try!(dc.stream.read_exact_(&mut rbuf[0..bytes_pp]));
                let mut p = gotten;
                while p < gotten+copysize {
                    copy_memory(&rbuf[0..bytes_pp], &mut src_line[p..p+bytes_pp]);
                    p += bytes_pp;
                }
            } else {    // it's raw
                try!(dc.stream.read_exact_(&mut src_line[gotten..gotten+copysize]));
            }
            wanted -= copysize;
            plen -= copysize;
        }

        convert(&src_line[..], &mut result[ti as usize .. (ti+tgt_linesz) as usize],
                c0, c1, c2, c3);
        ti += tgt_stride;
    }

    Ok(result)
}

struct TgaDecoder<'r, R:'r> {
    stream        : &'r mut R,
    w             : usize,
    h             : usize,
    origin_at_top : bool,
    rle           : bool,          // run length compressed
    src_fmt       : ColFmt,
    tgt_fmt       : ColFmt,
}

enum TgaDataType {
    Idx          = 1,
    TrueColor    = 2,
    Gray         = 3,
    IdxRLE       = 9,
    TrueColorRLE = 10,
    GrayRLE      = 11,
}

impl TgaDataType {
    fn from_u8(val: u8) -> Option<TgaDataType> {
        match val {
            1 => Some(TgaDataType::Idx),
            2 => Some(TgaDataType::TrueColor),
            3 => Some(TgaDataType::Gray),
            9 => Some(TgaDataType::IdxRLE),
            10 => Some(TgaDataType::TrueColorRLE),
            11 => Some(TgaDataType::GrayRLE),
            _ => None,
        }
    }
}

// --------------------------------------------------
// TGA Encoder

/// Writes an image and converts it to requested color type.
pub fn write<W: Write>(writer: &mut W, w: usize, h: usize, src_fmt: ColFmt, data: &[u8],
                                                                      tgt_type: ColType,
                                                              src_stride: Option<usize>)
                                                                        -> ::Result<()>
{
    if src_fmt == ColFmt::Auto { return Err(::Error::InvalidArg("invalid format")) }
    let stride = src_stride.unwrap_or(w * src_fmt.channels());

    if w < 1 || h < 1 || 0xffff < w || 0xffff < h
    || (src_stride.is_none() && src_fmt.channels() * w * h != data.len())
    || (src_stride.is_some() && data.len() < stride * (h-1) + w * src_fmt.channels()) {
        return Err(::Error::InvalidArg("invalid dimensions or data length"))
    }

    let tgt_fmt = match tgt_type {
        ColType::Gray       => ColFmt::Y,
        ColType::GrayAlpha  => ColFmt::YA,
        ColType::Color      => ColFmt::BGR,
        ColType::ColorAlpha => ColFmt::BGRA,
        ColType::Auto => match src_fmt {
            ColFmt::Y                 => ColFmt::Y,
            ColFmt::YA   | ColFmt::AY => ColFmt::YA,
            ColFmt::RGB  | ColFmt::BGR => ColFmt::BGR,
            ColFmt::RGBA | ColFmt::BGRA => ColFmt::BGRA,
            ColFmt::ARGB | ColFmt::ABGR => ColFmt::BGRA,
            ColFmt::Auto => return Err(::Error::InvalidArg("invalid format")),
        },
    };

    let ec = &mut TgaEncoder {
        stream    : writer,
        w         : w,
        h         : h,
        src_stride: stride,
        src_fmt   : src_fmt,
        tgt_fmt   : tgt_fmt,
        rle       : true,
        data      : data,
    };

    try!(write_header(ec));
    try!(write_image_data(ec));

    // write footer
    let ftr: &'static [u8] =
        b"\x00\x00\x00\x00\
          \x00\x00\x00\x00\
          TRUEVISION-XFILE.\x00";
    try!(ec.stream.write_all(ftr));

    try!(ec.stream.flush());
    Ok(())
}

fn write_header<W: Write>(ec: &mut TgaEncoder<W>) -> ::Result<()> {
    use self::TgaDataType::*;
    let (data_type, has_alpha) = match ec.tgt_fmt.channels() {
        1 => (if ec.rle { GrayRLE      } else { Gray      }, false),
        2 => (if ec.rle { GrayRLE      } else { Gray      }, true),
        3 => (if ec.rle { TrueColorRLE } else { TrueColor }, false),
        4 => (if ec.rle { TrueColorRLE } else { TrueColor }, true),
        _ => return Err(::Error::Internal("wrong format"))
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
        (ec.tgt_fmt.channels() * 8) as u8,
        if has_alpha {8u8} else {0u8},  // flags
    ];

    try!(ec.stream.write_all(hdr));
    Ok(())
}

fn write_image_data<W: Write>(ec: &mut TgaEncoder<W>) -> ::Result<()> {
    let src_linesz = ec.w * ec.src_fmt.channels();
    let tgt_linesz = ec.w * ec.tgt_fmt.channels();
    let mut tgt_line = vec![0u8; tgt_linesz];
    let mut si = ec.h as usize * ec.src_stride;

    let (convert, c0, c1, c2, c3) = try!(converter(ec.src_fmt, ec.tgt_fmt));

    if !ec.rle {
        for _ in (0 .. ec.h) {
            si -= ec.src_stride; // origin at bottom
            convert(&ec.data[si..si+src_linesz], &mut tgt_line[..],
                    c0, c1, c2, c3);
            try!(ec.stream.write_all(&tgt_line[..]));
        }
        return Ok(());
    }

    // ----- RLE -----

    let max_packets_per_line = (tgt_linesz+127) / 128;
    let mut cmp_buf = vec![0u8; tgt_linesz+max_packets_per_line];
    for _ in (0 .. ec.h) {
        si -= ec.src_stride;
        convert(&ec.data[si .. si+src_linesz], &mut tgt_line[..],
                c0, c1, c2, c3);
        let compressed_line =
            rle_compress(&tgt_line[..], &mut cmp_buf[..], ec.w, ec.tgt_fmt.channels());
        try!(ec.stream.write_all(&compressed_line[..]));
    }
    return Ok(());
}

fn rle_compress<'a>(line: &[u8], cmp_buf: &'a mut[u8], w: usize, bytes_pp: usize)
                                                                    -> &'a [u8]
{
    let rle_limit = if 1 < bytes_pp { 2 } else { 3 };   // run len worth an RLE packet
    let mut rawlen = 0;
    let mut raw_i = 0;   // start of raw packet data
    let mut cmp_i = 0;
    let mut pixels_left = w;
    let mut px: &[u8];

    let mut i = bytes_pp;
    while 0 < pixels_left {
        let mut runlen = 1;
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
                    &line[raw_i..raw_i+copysize],
                    &mut cmp_buf[cmp_i..cmp_i+copysize]
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
                    &line[raw_i..raw_i+copysize],
                    &mut cmp_buf[cmp_i..cmp_i+copysize]
                );
                cmp_i += copysize;
                rawlen = 0;
            }

            // store RLE packet
            cmp_buf[cmp_i] = (0x80 | (runlen-1)) as u8;   // packet header
            cmp_i += 1;
            copy_memory(
                &px[0..bytes_pp],
                &mut cmp_buf[cmp_i..cmp_i+bytes_pp]
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
            &line[raw_i..raw_i+copysize],
            &mut cmp_buf[cmp_i..cmp_i+copysize]
        );
        cmp_i += copysize;
    }

    &cmp_buf[0 .. cmp_i]
}

struct TgaEncoder<'r, R:'r> {
    stream        : &'r mut R,
    w             : usize,
    h             : usize,
    src_stride    : usize,
    tgt_fmt       : ColFmt,
    src_fmt       : ColFmt,
    rle           : bool,          // run length compressed
    data          : &'r [u8],
}
