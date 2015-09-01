// Copyright (c) 2015 Tero Hänninen, license: MIT

use std::io::{self, Read, Seek, SeekFrom};
use super::{
    Image, Info, ColFmt, ColType, error,
    copy_memory, converter,
    u32_from_le, i32_from_le, u16_from_le, IFRead,
};

/// Returns width, height and color type of the image.
pub fn read_info<R: Read+Seek>(reader: &mut R) -> io::Result<Info> {
    let hdr = try!(read_header(reader));

    Ok(Info {
        w: hdr.width.abs() as usize,
        h: hdr.height.abs() as usize,
        ct: if let Some(mask) = hdr.dib_v3_alpha_mask {
               if mask != 0 { ColType::ColorAlpha } else { ColType::Color }
           } else {
               ColType::Color
           },
    })
}

/// Header of a BMP image.
#[derive(Debug)]
struct BmpHeader {
    // BMP
    pub file_size             : u32,
    pub pixel_data_offset     : usize,

    // DIB
    pub dib_size              : usize,
    pub width                 : isize,
    pub height                : isize,
    pub planes                : u16,
    pub dib_v1                : Option<DibV1>,
    pub dib_v2                : Option<DibV2>,
    pub dib_v3_alpha_mask     : Option<u32>,
    pub dib_v4                : Option<DibV4>,
    pub dib_v5                : Option<DibV5>,
}

/// Optional part of a BMP header.
#[derive(Debug)]
struct DibV1 {
    pub bits_pp               : usize,
    pub compression           : u32,
    pub idat_size             : usize,
    pub pixels_per_meter_x    : usize,
    pub pixels_per_meter_y    : usize,
    pub palette_length        : usize,    // colors in color table
    pub important_color_count : u32,
}

/// Optional part of a BMP header.
#[derive(Debug)]
struct DibV2 {
    pub red_mask              : u32,
    pub green_mask            : u32,
    pub blue_mask             : u32,
}

/// Optional part of a BMP header.
#[derive(Debug)]
struct DibV4 {
    pub color_space_type      : u32,
    pub color_space_endpoints : Vec<u8>,//[u8; 36],     // Vec for Debug
    pub gamma_red             : u32,
    pub gamma_green           : u32,
    pub gamma_blue            : u32,
}

/// Optional part of a BMP header.
#[derive(Debug)]
struct DibV5 {
    pub icc_profile_data      : u32,
    pub icc_profile_size      : u32,
}

/// Reads a BMP header.
fn read_header<R: Read+Seek>(reader: &mut R) -> io::Result<BmpHeader> {
    let mut bmp_header = [0u8; 18]; // bmp header + size of dib header
    try!(reader.read_exact_(&mut bmp_header[..]));

    if &bmp_header[0..2] != [0x42, 0x4d] {
        return error("corrupt bmp header");
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
        _ => return error("unsupported dib version"),
    };
    let mut dib_header = vec![0u8; dib_size-4];
    try!(reader.read_exact_(&mut dib_header[..]));

    Ok(BmpHeader {
        file_size             : u32_from_le(&bmp_header[2..6]),
        pixel_data_offset     : u32_from_le(&bmp_header[10..14]) as usize,
        width                 : i32_from_le(&dib_header[0..4]) as isize,
        height                : i32_from_le(&dib_header[4..8]) as isize,
        planes                : u16_from_le(&dib_header[8..10]),
        dib_size              : dib_size,
        dib_v1:
            if 1 <= dib_version {
                Some(DibV1 {
                    bits_pp               : u16_from_le(&dib_header[10..12]) as usize,
                    compression           : u32_from_le(&dib_header[12..16]),
                    idat_size             : u32_from_le(&dib_header[16..20]) as usize,
                    pixels_per_meter_x    : u32_from_le(&dib_header[20..24]) as usize,
                    pixels_per_meter_y    : u32_from_le(&dib_header[24..28]) as usize,
                    palette_length        : u32_from_le(&dib_header[28..32]) as usize,
                    important_color_count : u32_from_le(&dib_header[32..36]),
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
        dib_v4:
            if 4 <= dib_version {
                let mut color_space_endpoints = Vec::with_capacity(36);//[0u8; 36];
                color_space_endpoints.extend((&dib_header[56..92]).iter().map(|&b| b));
                //copy_memory(&dib_header[56..92], &mut color_space_endpoints[..]);
                Some(DibV4 {
                    color_space_type      : u32_from_le(&dib_header[52..56]),
                    color_space_endpoints : color_space_endpoints,
                    gamma_red             : u32_from_le(&dib_header[92..96]),
                    gamma_green           : u32_from_le(&dib_header[96..100]),
                    gamma_blue            : u32_from_le(&dib_header[100..104]),
                })
            } else {
                None
            },
        dib_v5:
            if 5 <= dib_version {
                Some(DibV5 {
                    icc_profile_data      : u32_from_le(&dib_header[108..112]),
                    icc_profile_size      : u32_from_le(&dib_header[112..116]),
                })
            } else {
                None
            }
    })
}

pub fn detect<R: Read+Seek>(reader: &mut R) -> bool {
    let mut bmp_header = [0u8; 18]; // bmp header + size of dib header
    let result =
        reader.read_exact_(&mut bmp_header[..]).is_ok()
        && &bmp_header[0..2] == [0x42, 0x4d]
        && match u32_from_le(&bmp_header[14..18]) {
            12 | 40 | 52 | 56 | 108 | 124 => true,
            _ => false,
        };
    let _ = reader.seek(SeekFrom::Start(0));
    result
}

const CMP_RGB: u32        = 0;
const CMP_BITS: u32       = 3;
//const CMP_ALPHA_BITS: u32 = 6;

/// Reads an image and converts it to requested format.
///
/// Passing `ColFmt::Auto` as req_fmt converts the data to `RGB` or `RGBA`. The DIB
/// headers BITMAPV4HEADER and BITMAPV5HEADER are ignored if present.
pub fn read<R: Read+Seek>(reader: &mut R, req_fmt: ColFmt) -> io::Result<Image> {
    let hdr = try!(read_header(reader));

    if hdr.width < 1 || hdr.height == 0 { return error("invalid dimensions") }
    if hdr.pixel_data_offset < (14 + hdr.dib_size)
    || hdr.pixel_data_offset > 0xffffff /* arbitrary */ {
        return error("invalid pixel data offset")
    }
    if hdr.planes != 1 { return error("not supported") }

    let (bytes_pp, paletted, palette_length, rgb_masked) =
        if let Some(ref dv1) = hdr.dib_v1 {
            if 256 < dv1.palette_length { return error("ivnalid palette length") }
            if dv1.bits_pp <= 8
            && (dv1.palette_length == 0 || dv1.compression != CMP_RGB) {
                 return error("invalid format")
            }
            if dv1.compression != CMP_RGB && dv1.compression != CMP_BITS {
                 return error("unsupported compression")
            }
            let rgb_masked = dv1.compression == CMP_BITS;

            match dv1.bits_pp {
                8   => (1, true,  dv1.palette_length, rgb_masked),
                24  => (3, false, dv1.palette_length, rgb_masked),
                32  => (4, false, dv1.palette_length, rgb_masked),
                _   => return error("not supported (dv1)"),
            }
        } else {
            (1, true, 256, false)
        };
    let pe_fmt = if hdr.dib_v1.is_some() { ColFmt::BGRA } else { ColFmt::BGR };

    fn mask_to_idx(mask: u32) -> io::Result<usize> {
        match mask {
            0xff00_0000 => Ok(3),
            0x00ff_0000 => Ok(2),
            0x0000_ff00 => Ok(1),
            0x0000_00ff => Ok(0),
            _ => return error("unsupported mask")
        }
    }

    let (redi, greeni, bluei) = match (rgb_masked, hdr.dib_v2) {
        (true, Some(ref dv2)) => {
            (try!(mask_to_idx(dv2.red_mask)),
             try!(mask_to_idx(dv2.green_mask)),
             try!(mask_to_idx(dv2.blue_mask)))
        }
        (false, _) => { (2, 1, 0) }
        _ => return error("invalid format"),
    };

    let (alpha_masked, alphai) =
        if let Some(mask) = hdr.dib_v3_alpha_mask {
            if mask == 0 { (false, 0) } else { (true, try!(mask_to_idx(mask))) }
        } else {
            (false, 0)
        };

    let (palette, mut depaletted_line) =
        if paletted {
            let mut palette = vec![0u8; palette_length * pe_fmt.bytes_pp()];
            try!(reader.read_exact_(&mut palette[..]));
            (palette, vec![0u8; hdr.width as usize * pe_fmt.bytes_pp()])
        } else {
            (Vec::new(), Vec::new())
        };

    try!(reader.seek(SeekFrom::Start(hdr.pixel_data_offset as u64)));

    let tgt_fmt = {
        use super::ColFmt::*;
        match req_fmt {
            Y | YA | RGB | RGBA | BGR | BGRA => req_fmt,
            Auto => if alpha_masked { RGBA } else { RGB },
        }
    };

    let convert =
        try!(converter(if paletted { pe_fmt } else { ColFmt::BGRA }, tgt_fmt));

    let src_linesize = hdr.width as usize * bytes_pp;  // without padding
    let src_pad = 3 - ((src_linesize-1) % 4);
    let tgt_bytespp = tgt_fmt.bytes_pp();
    let tgt_linesize = (hdr.width as usize * tgt_bytespp) as isize;

    let (tgt_stride, mut ti): (isize, isize) =
        if hdr.height < 0 {
            (tgt_linesize, 0)
        } else {
            (-tgt_linesize, (hdr.height-1) * tgt_linesize)
        };

    let mut src_line_buf = vec![0u8; src_linesize + src_pad];
    let mut bgra_line_buf = vec![0u8; if paletted { 0 } else { hdr.width as usize * 4 }];
    let mut result =
        vec![0u8; hdr.width as usize * hdr.height.abs() as usize * tgt_bytespp];

    for _ in (0 .. hdr.height.abs()) {
        try!(reader.read_exact_(&mut src_line_buf[..]));
        let src_line = &src_line_buf[..src_linesize];

        if paletted {
            let ps = pe_fmt.bytes_pp();
            let mut di = 0;
            for &idx in src_line {
                if idx as usize > palette_length {
                    return error("invalid palette index");
                }
                let idx = idx as usize * ps;
                copy_memory(&palette[idx .. idx+ps], &mut depaletted_line[di .. di+ps]);
                if ps == 4 {
                    depaletted_line[di+3] = 255;
                }
                di += ps;
            }
            convert(&depaletted_line, &mut result[ti as usize..(ti+tgt_linesize) as usize]);
        } else {
            let mut si = 0;
            let mut di = 0;
            while si < src_line.len() {
                bgra_line_buf[di + 0] = src_line[si + bluei];
                bgra_line_buf[di + 1] = src_line[si + greeni];
                bgra_line_buf[di + 2] = src_line[si + redi];
                bgra_line_buf[di + 3] = if alpha_masked { src_line[si + alphai] }
                                                   else { 255 };
                si += bytes_pp;
                di += 4;
            }
            convert(&bgra_line_buf, &mut result[ti as usize..(ti+tgt_linesize) as usize]);
        }

        ti += tgt_stride;
    }

    Ok(Image {
        w   : hdr.width as usize,
        h   : hdr.height.abs() as usize,
        fmt : tgt_fmt,
        buf : result,
    })
}
