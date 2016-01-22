// Copyright (c) 2015 Tero Hänninen, license: MIT

use std::io::{Read, Write, Seek, SeekFrom};
use super::{
    Image, Info, ColFmt, ColType, copy_memory, converter,
    u32_from_le, i32_from_le, u16_from_le, u32_to_le, u16_to_le,
};

/// Returns width, height and color type of the image.
pub fn read_info<R: Read+Seek>(reader: &mut R) -> ::Result<Info> {
    let hdr = try!(read_header(reader));

    Ok(Info {
        w: hdr.width.abs() as usize,
        h: hdr.height.abs() as usize,
        ct: match (hdr.bits_pp, hdr.dib_v3_alpha_mask) {
                (32, Some(mask)) if mask != 0 => ColType::ColorAlpha,
                                            _ => ColType::Color,
        },
    })
}

/// Header of a BMP image.
struct BmpHeader {
    // BMP
    //file_size             : u32,
    pixel_data_offset     : usize,

    // DIB
    dib_size              : usize,
    width                 : isize,
    height                : isize,
    planes                : u16,
    bits_pp               : usize,
    dib_v1                : Option<DibV1>,
    dib_v2                : Option<DibV2>,
    dib_v3_alpha_mask     : Option<u32>,
    // dib_v4 and dib_v5 are ignored
}

struct DibV1 {
    compression           : u32,
    //idat_size             : usize,
    //pixels_per_meter_x    : usize,
    //pixels_per_meter_y    : usize,
    palette_length        : usize,    // colors in color table
    //important_color_count : u32,
}

struct DibV2 {
    red_mask              : u32,
    green_mask            : u32,
    blue_mask             : u32,
}

/// Reads a BMP header.
fn read_header<R: Read+Seek>(reader: &mut R) -> ::Result<BmpHeader> {
    let mut bmp_header = [0u8; 18]; // bmp header + size of dib header
    try!(reader.read_exact(&mut bmp_header[..]));

    if &bmp_header[0..2] != [0x42, 0x4d] {
        return Err(::Error::InvalidData("corrupt bmp header"))
    }

    // the value of dib_size is actually part of the dib header
    let dib_size = u32_from_le(&bmp_header[14..18]) as usize;
    let dib_version = match dib_size {
        12 => 0,
        40 => 1,
        52 => 2,
        56 => 3,
        108 => 4,
        124 => 5,
        _ => return Err(::Error::Unsupported("dib version")),
    };
    let mut dib_header = vec![0u8; dib_size-4];
    try!(reader.read_exact(&mut dib_header[..]));

    let (width, height, planes, bits_pp) =
        if dib_version == 0 {
            ( u16_from_le(&dib_header[0..2]) as isize
            , u16_from_le(&dib_header[2..4]) as isize
            , u16_from_le(&dib_header[4..6])
            , u16_from_le(&dib_header[6..8]) as usize)
        } else {
            ( i32_from_le(&dib_header[0..4]) as isize
            , i32_from_le(&dib_header[4..8]) as isize
            , u16_from_le(&dib_header[8..10])
            , u16_from_le(&dib_header[10..12]) as usize)
        };

    Ok(BmpHeader {
        //file_size             : u32_from_le(&bmp_header[2..6]),
        pixel_data_offset     : u32_from_le(&bmp_header[10..14]) as usize,
        width                 : width,
        height                : height,
        planes                : planes,
        bits_pp               : bits_pp,
        dib_size              : dib_size,
        dib_v1:
            if 1 <= dib_version {
                Some(DibV1 {
                    compression           : u32_from_le(&dib_header[12..16]),
                    //idat_size             : u32_from_le(&dib_header[16..20]) as usize,
                    //pixels_per_meter_x    : u32_from_le(&dib_header[20..24]) as usize,
                    //pixels_per_meter_y    : u32_from_le(&dib_header[24..28]) as usize,
                    palette_length        : u32_from_le(&dib_header[28..32]) as usize,
                    //important_color_count : u32_from_le(&dib_header[32..36]),
                })
            } else {
                None
            },
        dib_v2:
            if 2 <= dib_version {
                Some(DibV2 {
                    red_mask              : u32_from_le(&dib_header[36..40]),
                    green_mask            : u32_from_le(&dib_header[40..44]),
                    blue_mask             : u32_from_le(&dib_header[44..48]),
                })
            } else {
                None
            },
        dib_v3_alpha_mask:
            if 3 <= dib_version {
                Some(u32_from_le(&dib_header[48..52]))
            } else {
                None
            },
    })
}

pub fn detect<R: Read+Seek>(reader: &mut R) -> bool {
    let mut bmp_header = [0u8; 18]; // bmp header + size of dib header
    let start = match reader.seek(SeekFrom::Current(0))
        { Ok(s) => s, Err(_) => return false };
    let result =
        reader.read_exact(&mut bmp_header[..]).is_ok()
        && &bmp_header[0..2] == [0x42, 0x4d]
        && match u32_from_le(&bmp_header[14..18]) {
            12 | 40 | 52 | 56 | 108 | 124 => true,
            _ => false,
        };
    let _ = reader.seek(SeekFrom::Start(start));
    result
}

const CMP_RGB: u32        = 0;
const CMP_BITS: u32       = 3;
//const CMP_ALPHA_BITS: u32 = 6;

/// Reads an image and converts it to requested format.
///
/// Passing `ColFmt::Auto` as req_fmt converts the data to `RGB` or `RGBA`. The DIB
/// headers BITMAPV4HEADER and BITMAPV5HEADER are ignored if present.
pub fn read<R: Read+Seek>(reader: &mut R, req_fmt: ColFmt) -> ::Result<Image<u8>> {
    let hdr = try!(read_header(reader));

    if hdr.width < 1 || hdr.height == 0 {
        return Err(::Error::InvalidData("invalid dimensions"))
    }
    if hdr.pixel_data_offset < (14 + hdr.dib_size)
    || hdr.pixel_data_offset > 0xffffff /* arbitrary */ {
        return Err(::Error::InvalidData("invalid pixel data offset"))
    }
    if hdr.planes != 1 { return Err(::Error::Unsupported("planes != 1")) }

    let (bytes_pp, paletted, palette_length, rgb_masked) =
        if let Some(ref dv1) = hdr.dib_v1 {
            if 256 < dv1.palette_length {
                return Err(::Error::InvalidData("palette length"))
            }
            if hdr.bits_pp <= 8
            && (dv1.palette_length == 0 || dv1.compression != CMP_RGB) {
                 return Err(::Error::InvalidData("invalid format"))
            }
            if dv1.compression != CMP_RGB && dv1.compression != CMP_BITS {
                 return Err(::Error::Unsupported("compression"))
            }
            let rgb_masked = dv1.compression == CMP_BITS;

            match hdr.bits_pp {
                8   => (1, true,  dv1.palette_length, rgb_masked),
                24  => (3, false, dv1.palette_length, rgb_masked),
                32  => (4, false, dv1.palette_length, rgb_masked),
                _   => return Err(::Error::Unsupported("bit depth")),
            }
        } else {
            (1, true, 256, false)
        };
    let pe_fmt = if hdr.dib_v1.is_some() { ColFmt::BGRA } else { ColFmt::BGR };

    fn mask_to_idx(mask: u32) -> ::Result<usize> {
        match mask {
            0xff00_0000 => Ok(3),
            0x00ff_0000 => Ok(2),
            0x0000_ff00 => Ok(1),
            0x0000_00ff => Ok(0),
            _ => return Err(::Error::Unsupported("channel mask"))
        }
    }

    let (redi, greeni, bluei) = match (rgb_masked, hdr.dib_v2) {
        (true, Some(ref dv2)) => {
            (try!(mask_to_idx(dv2.red_mask)),
             try!(mask_to_idx(dv2.green_mask)),
             try!(mask_to_idx(dv2.blue_mask)))
        }
        (false, _) => { (2, 1, 0) }
        _ => return Err(::Error::InvalidData("invalid format")),
    };

    let (alpha_masked, alphai) =
        match (bytes_pp, hdr.dib_v3_alpha_mask) {
            (4, Some(mask)) if mask != 0 => (true, try!(mask_to_idx(mask))),
                                       _ => (false, 0),
        };

    let (palette, mut depaletted) =
        if paletted {
            let mut palette = vec![0u8; palette_length * pe_fmt.channels()];
            try!(reader.read_exact(&mut palette[..]));
            (palette, vec![0u8; hdr.width as usize * pe_fmt.channels()])
        } else {
            (Vec::new(), Vec::new())
        };

    try!(reader.seek(SeekFrom::Start(hdr.pixel_data_offset as u64)));

    let tgt_fmt = {
        use super::ColFmt::*;
        match req_fmt {
            Auto => if alpha_masked { RGBA } else { RGB },
               _ => req_fmt,
        }
    };

    let (convert, c0, c1, c2, c3) =
        try!(converter(if paletted { pe_fmt } else { ColFmt::BGRA }, tgt_fmt));

    let src_linesz = hdr.width as usize * bytes_pp;  // without padding
    let src_pad = 3 - ((src_linesz-1) % 4);
    let tgt_bytespp = tgt_fmt.channels();
    let tgt_linesz = (hdr.width as usize * tgt_bytespp) as isize;

    let (tgt_stride, mut ti): (isize, isize) =
        if hdr.height < 0 {
            (tgt_linesz, 0)
        } else {
            (-tgt_linesz, (hdr.height-1) * tgt_linesz)
        };

    let mut src_line = vec![0u8; src_linesz + src_pad];
    let mut bgra_line = vec![0u8; if paletted { 0 } else { hdr.width as usize * 4 }];
    let mut result =
        vec![0u8; hdr.width as usize * hdr.height.abs() as usize * tgt_bytespp];

    for _ in 0 .. hdr.height.abs() {
        try!(reader.read_exact(&mut src_line[..]));
        let src_line = &src_line[..src_linesz];

        if paletted {
            let ps = pe_fmt.channels();
            let mut di = 0;
            for &idx in src_line {
                if idx as usize > palette_length {
                    return Err(::Error::InvalidData("palette index"));
                }
                let idx = idx as usize * ps;
                copy_memory(&palette[idx .. idx+ps], &mut depaletted[di .. di+ps]);
                if ps == 4 {
                    depaletted[di+3] = 255;
                }
                di += ps;
            }
            convert(&depaletted, &mut result[ti as usize..(ti+tgt_linesz) as usize],
                    c0, c1, c2, c3);
        } else {
            let mut si = 0;
            let mut di = 0;
            while si < src_line.len() {
                bgra_line[di + 0] = src_line[si + bluei];
                bgra_line[di + 1] = src_line[si + greeni];
                bgra_line[di + 2] = src_line[si + redi];
                bgra_line[di + 3] = if alpha_masked { src_line[si + alphai] } else { 255 };
                si += bytes_pp;
                di += 4;
            }
            convert(&bgra_line, &mut result[ti as usize..(ti+tgt_linesz) as usize],
                    c0, c1, c2, c3);
        }

        ti += tgt_stride;
    }

    Ok(Image::<u8> {
        w   : hdr.width as usize,
        h   : hdr.height.abs() as usize,
        fmt : tgt_fmt,
        buf : result,
    })
}

// --------------------------------------------------
// BMP encoder

/// Writes an image and converts it to requested color type (grayscale not supported).
pub fn write<W: Write>(writer: &mut W, w: usize, h: usize, src_fmt: ColFmt, data: &[u8],
                                                                      tgt_type: ColType,
                                                              src_stride: Option<usize>)
                                                                        -> ::Result<()>
{
    if src_fmt == ColFmt::Auto { return Err(::Error::InvalidArg("invalid format")) }
    let stride = src_stride.unwrap_or(w * src_fmt.channels());

    if w < 1 || h < 1 || 0x7fff < w || 0x7fff < h
    || (src_stride.is_none() && src_fmt.channels() * w * h != data.len())
    || (src_stride.is_some() && data.len() < stride * (h-1) + w * src_fmt.channels()) {
        return Err(::Error::InvalidArg("dimensions or data length"))
    }

    let tgt_fmt = match tgt_type {
        ColType::Color      => ColFmt::BGR,
        ColType::ColorAlpha => ColFmt::BGRA,
        ColType::Auto => if src_fmt.has_alpha() == Some(true) {
                             ColFmt::BGRA
                         } else {
                             ColFmt::BGR
                         },
        _ => return Err(::Error::InvalidArg("unsupported target color type")),
    };

    let dib_size = 108;
    let tgt_linesz = w * tgt_fmt.channels();
    let pad = 3 - ((tgt_linesz-1) & 3);
    let idat_offset = 14 + dib_size;       // bmp file header + dib header
    let filesize = idat_offset + h * (tgt_linesz + pad);
    if filesize > 0xffff_ffff {
        return Err(::Error::InvalidData("image too large"))
    }

    let tgt_has_alpha = tgt_fmt.has_alpha() == Some(true);

    try!(writer.write_all(b"BM"));
    try!(writer.write_all(&u32_to_le(filesize as u32)[..]));
    try!(writer.write_all(&[0u8; 4]));                      // reserved
    try!(writer.write_all(&u32_to_le(idat_offset as u32)[..])); // offset of pixel data
    try!(writer.write_all(&u32_to_le(dib_size as u32)[..]));    // dib header size
    try!(writer.write_all(&u32_to_le(w as u32)[..]));
    try!(writer.write_all(&u32_to_le(h as u32)[..]));       // positive -> bottom-up
    try!(writer.write_all(&u16_to_le(1)[..]));              // planes
    try!(writer.write_all(&u16_to_le((tgt_fmt.channels() * 8) as u16)[..])); // bpp
    try!(writer.write_all(&u32_to_le(if tgt_has_alpha { CMP_BITS } else { CMP_RGB })));
    try!(writer.write_all(&[0u8; 5 * 4]));   // rest of DibV1
    if tgt_has_alpha {
        // red, green, blue and alpha masks
        try!(writer.write_all(&[0, 0, 0xff, 0,
                                0, 0xff, 0, 0,
                                0xff, 0, 0, 0,
                                0, 0, 0, 0xff]));
    } else {
        try!(writer.write_all(&[0u8; 4 * 4]));   // DibV2 & DibV3
    }
    try!(writer.write_all(b"BGRs"));
    try!(writer.write_all(&[0u8; 12 * 4]));   // rest of DibV4

    let (convert, c0, c1, c2, c3) = try!(converter(src_fmt, tgt_fmt));

    let mut tgt_line = vec![0u8; tgt_linesz + pad];
    let src_linesz = w * src_fmt.channels();
    let mut si = h * stride;

    for _ in 0 .. h {
        si -= stride;
        convert(&data[si .. si + src_linesz], &mut tgt_line[..tgt_linesz],
                c0, c1, c2, c3);
        try!(writer.write_all(&tgt_line));
    }

    try!(writer.flush());
    Ok(())
}
