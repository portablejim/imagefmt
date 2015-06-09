// Copyright (c) 2014-2015 Tero Hänninen, license: MIT

use std::io::{self, Read};
use std::iter::{repeat};
use std::mem::{zeroed};
use super::{
    Image, Info, ColFmt, ColType, error,
    copy_memory, u16_from_be, IFRead,
};

// Baseline JPEG decoder

/// Returns width, height and color type of the image.
pub fn read_jpeg_info<R: Read>(reader: &mut R) -> io::Result<Info> {
    let mut marker = [0u8; 2];

    // SOI
    try!(reader.read_exact(&mut marker));
    if &marker[0..2] != &[0xff, 0xd8_u8][..] {
        return error("not JPEG");
    }

    loop {
        try!(reader.read_exact(&mut marker));

        if marker[0] != 0xff { return error("no marker"); }
        while marker[1] == 0xff {
            try!(reader.read_exact(&mut marker[1..2]));
        }

        match marker[1] {
            SOF0 | SOF2 => {
                let mut tmp: [u8; 8] = [0,0,0,0, 0,0,0,0];
                try!(reader.read_exact(&mut tmp));
                return Ok(Info {
                    w: u16_from_be(&tmp[5..7]) as usize,
                    h: u16_from_be(&tmp[3..5]) as usize,
                    c: match tmp[7] {
                           1 => ColType::Gray,
                           3 => ColType::Color,
                           _ => return error("not valid baseline jpeg")
                       },
                });
            }
            SOS | EOI => return error("no frame header"),
            DRI | DHT | DQT | COM | APP0 ... APPF => {
                let mut tmp: [u8; 2] = [0, 0];
                try!(reader.read_exact(&mut tmp));
                let len = u16_from_be(&mut tmp[..]) - 2;
                try!(reader.skip(len as usize));
            }
            _ => return error("invalid / unsupported marker"),
        }
    }
}

/// Reads an image and converts it to requested format.
///
/// Passing `ColFmt::Auto` as `req_fmt` converts the data to `Y` or `RGB`.
pub fn read_jpeg<R: Read>(reader: &mut R, req_fmt: ColFmt) -> io::Result<Image> {
    use super::ColFmt::*;
    let req_fmt = match req_fmt {
        Auto | Y | YA | RGB | RGBA => req_fmt,
        _ => return error("format not supported")
    };

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
        return error("no image data");
    }
    dc.tgt_fmt =
        if req_fmt == ColFmt::Auto {
            match dc.num_comps {
                1 => ColFmt::Y, 3 => ColFmt::RGB,
                _ => return error("internal error")
            }
        } else {
            req_fmt
        };

    for comp in dc.comps.iter_mut() {
        comp.data = repeat(0u8).take(dc.num_mcu_x*comp.sfx*8*dc.num_mcu_y*comp.sfy*8).collect();
    }

    Ok(Image {
        w      : dc.w,
        h      : dc.h,
        fmt    : dc.tgt_fmt,
        pixels : {
            // progressive images aren't supported so only one scan
            try!(decode_scan(dc));
            // throw away fill samples and convert to target format
            try!(reconstruct(dc))
        }
    })
}

fn read_app0<R: Read>(reader: &mut R) -> io::Result<()> {
    let mut buf = [0u8; 16];
    try!(reader.read_exact(&mut buf));

    let len = u16_from_be(&buf[0..2]) as usize;

    if &buf[2..7] != b"JFIF\0" || len < 16 {
        return error("not JPEG/JFIF");
    }

    if buf[7] != 1 {
        return error("version not supported");
    }

    // ignore density_unit, -x, -y at 13, 14..16, 16..18

    let thumbsize = buf[14] as usize * buf[15] as usize * 3;
    if thumbsize != len - 16 {
        return error("corrupt app0 marker");
    }
    reader.skip(thumbsize)
}

struct JpegDecoder<'r, R: Read + 'r> {
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

fn read_markers<R: Read>(dc: &mut JpegDecoder<R>) -> io::Result<()> {
    let mut marker = [0u8; 2];
    // SOI
    try!(dc.stream.read_exact(&mut marker));
    if &marker[0..2] != &[0xff, 0xd8_u8][..] {
        return error("not JPEG");
    }

    let mut has_next_scan_header = false;
    while !has_next_scan_header && !dc.eoi_reached {
        try!(dc.stream.read_exact(&mut marker));

        if marker[0] != 0xff { return error("no marker"); }
        while marker[1] == 0xff {
            try!(dc.stream.read_exact(&mut marker[1..2]));
        }

        //println!("marker: 0x{:x}", marker[1]);
        match marker[1] {
            DHT => try!(read_huffman_tables(dc)),
            DQT => try!(read_quantization_tables(dc)),
            SOF0 => {
                if dc.has_frame_header {
                    return error("extra frame header");
                }
                try!(read_frame_header(dc));
                dc.has_frame_header = true;
            }
            SOS => {
                if !dc.has_frame_header {
                    return error("no frame header");
                }
                try!(read_scan_header(dc));
                has_next_scan_header = true;
            }
            DRI => try!(read_restart_interval(dc)),
            EOI => dc.eoi_reached = true,
            APP0 => try!(read_app0(dc.stream)),
            APP1 ... APPF | COM => {
                let mut tmp = [0u8; 2];
                try!(dc.stream.read_exact(&mut tmp));
                let len = u16_from_be(&mut tmp[..]);
                if len < 2 { return error("invalid data length") }
                try!(dc.stream.skip(len as usize - 2));
            }
            SOF2 => return error("progressive jpeg not supported"),
            _ => return error("unsupported marker"),
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
const APP0: u8 = 0xe0;    // application 0 segment (jfif)
const APP1: u8 = 0xe1;    // application 1 segment (exif)
// ...
const APPF: u8 = 0xef;    // application f segment
//const DAC: u8 = 0xcc;     // define arithmetic conditioning table
const COM: u8 = 0xfe;     // comment
const EOI: u8 = 0xd9;     // end of image

fn read_huffman_tables<R: Read>(dc: &mut JpegDecoder<R>) -> io::Result<()> {
    let mut buf = [0u8; 17];
    try!(dc.stream.read_exact(&mut buf[0..2]));
    let mut len = u16_from_be(&buf[0..2]) as isize -2;

    while 0 < len {
        try!(dc.stream.read_exact(&mut buf[0..17]));  // info byte and BITS
        let table_class = buf[0] >> 4;            // 0 = dc table, 1 = ac table
        let table_slot = (buf[0] & 0xf) as usize;  // must be 0 or 1 for baseline
        if 1 < table_slot || 1 < table_class {
            return error("invalid / not supported");
        }

        // compute total number of huffman codes
        let mut mt = 0;
        for i in (1..17) {
            mt += buf[i] as usize;
        }
        if 256 < mt {
            return error("invalid / not supported");
        }

        if table_class == 0 {
            try!(dc.stream.read_exact(&mut dc.dc_tables[table_slot].values[0..mt]));
            derive_table(&mut dc.dc_tables[table_slot], &buf[1..17]);
        } else {
            try!(dc.stream.read_exact(&mut dc.ac_tables[table_slot].values[0..mt]));
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
    for i in (0..16) {
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
    for i in (0..16) {
        mincode[i] = -1;
        maxcode[i] = -1;
        valptr[i] = -1;
    }

    let mut j = 0;
    for i in (0..16) {
        if num_values[i] != 0 {
            valptr[i] = j as i16;
            mincode[i] = codes[j];
            j += (num_values[i] - 1) as usize;
            maxcode[i] = codes[j];
            j += 1;
        }
    }
}

fn read_quantization_tables<R: Read>(dc: &mut JpegDecoder<R>) -> io::Result<()> {
    let mut buf = [0u8; 2];
    try!(dc.stream.read_exact(&mut buf[0..2]));
    let mut len = u16_from_be(&buf[0..2]) as usize -2;
    if len % 65 != 0 {
        return error("invalid / not supported");
    }

    while 0 < len {
        try!(dc.stream.read_exact(&mut buf[0..1]));
        let precision = buf[0] >> 4;  // 0 = 8 bit, 1 = 16 bit
        let table_slot = (buf[0] & 0xf) as usize;
        if 3 < table_slot || precision != 0 {   // only 8 bit for baseline
            return error("invalid / not supported");
        }
        try!(dc.stream.read_exact(&mut dc.qtables[table_slot][0..64]));
        len -= 65;
    }
    Ok(())
}

fn read_frame_header<R: Read>(dc: &mut JpegDecoder<R>) -> io::Result<()> {
    let mut buf = [0u8; 9];
    try!(dc.stream.read_exact(&mut buf[0..8]));
    let len = u16_from_be(&buf[0..2]) as usize;
    let precision = buf[2];
    dc.h = u16_from_be(&buf[3..5]) as usize;
    dc.w = u16_from_be(&buf[5..7]) as usize;
    dc.num_comps = buf[7] as usize;

    if precision != 8 || (dc.num_comps != 1 && dc.num_comps != 3) ||
       len != 8 + dc.num_comps*3 {
        return error("invalid / not supported");
    }

    dc.hmax = 0;
    dc.vmax = 0;
    let mut mcu_du = 0; // data units in one mcu
    try!(dc.stream.read_exact(&mut buf[0 .. dc.num_comps*3]));

    for i in (0 .. dc.num_comps) {
        let ci = (buf[i*3]-1) as usize;
        if dc.num_comps <= ci {
            return error("invalid / not supported");
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
            return error("invalid / not supported");
        }

        if dc.hmax < comp.sfx { dc.hmax = comp.sfx; }
        if dc.vmax < comp.sfy { dc.vmax = comp.sfy; }
        mcu_du += comp.sfx * comp.sfy;
    }
    if 10 < mcu_du { return error("invalid / not supported"); }

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

fn read_scan_header<R: Read>(dc: &mut JpegDecoder<R>) -> io::Result<()> {
    let mut buf: [u8; 3] = [0, 0, 0];
    try!(dc.stream.read_exact(&mut buf));
    let len = u16_from_be(&buf[0..2]) as usize;
    let num_scan_comps = buf[2] as usize;

    if num_scan_comps != dc.num_comps || len != (6+num_scan_comps*2) {
        return error("invalid / not supported");
    }

    let mut compbuf: Vec<u8> = repeat(0u8).take(len-3).collect();
    try!(dc.stream.read_exact(&mut compbuf));

    for i in (0 .. num_scan_comps) {
        let comp_id = compbuf[i*2];
        let mut ci = 0;
        while ci < dc.num_comps && dc.comps[ci].id != comp_id { ci+=1 }
        if dc.num_comps <= ci {
            return error("invalid / not supported");
        }

        let tables = compbuf[i*2+1];
        dc.comps[ci].dc_table = (tables >> 4) as usize;
        dc.comps[ci].ac_table = (tables & 0xf) as usize;
        if 1 < dc.comps[ci].dc_table || 1 < dc.comps[i].ac_table {
            return error("invalid / not supported");
        }
    }

    // ignore last 3 bytes: spectral_start, spectral_end, approx
    Ok(())
}

fn read_restart_interval<R: Read>(dc: &mut JpegDecoder<R>) -> io::Result<()> {
    let mut buf: [u8; 4] = [0, 0, 0, 0];
    try!(dc.stream.read_exact(&mut buf));
    let len = u16_from_be(&buf[0..2]) as usize;
    if len != 4 { return error("invalid / not supported"); }
    dc.restart_interval = u16_from_be(&buf[2..4]) as usize;
    Ok(())
}

fn decode_scan<R: Read>(dc: &mut JpegDecoder<R>) -> io::Result<()> {
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
                            stbi_idct_block(dst, dst_stride, &data[..]);
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

fn read_restart<R: Read>(stream: &mut R) -> io::Result<()> {
    let mut buf: [u8; 2] = [0, 0];
    try!(stream.read_exact(&mut buf));
    if buf[0] != 0xff || buf[1] < RST0 || RST7 < buf[1] {
        return error("reset marker missing");
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
//fn decode_block<R: Read>(dc: &mut JpegDecoder<R>, comp: &mut Component,
//                                                     qtable: &[u8, ..64])
//                                                 -> io::Result<[i16, ..64]>
fn decode_block<R: Read>(dc: &mut JpegDecoder<R>, comp_idx: usize, qtable_idx: usize)
                                                             -> io::Result<[i16; 64]>
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

    let mut k = 1;
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
            return error("corrupt block");
        }
        res[DEZIGZAG[k] as usize] =
            (try!(receive_and_extend(dc, ssss)) * dc.qtables[qtable_idx][k] as isize) as i16;
        k += 1;
    }

    Ok(res)
}

//fn decode_huff<R: Read>(dc: &mut JpegDecoder<R>, tab: &HuffTab) -> io::Result<u8> {
fn decode_huff<R: Read>(dc: &mut JpegDecoder<R>, tab_idx: usize, dctab: bool) -> io::Result<u8> {
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
            return error("corrupt huffman coding");
        }
    }
    let j = (tab.valptr[i] - tab.mincode[i] + code) as usize;
    if tab.values.len() <= j {
        return error("corrupt huffman coding")
    }
    Ok(tab.values[j])
}

fn receive_and_extend<R: Read>(dc: &mut JpegDecoder<R>, s: u8) -> io::Result<isize> {
    // receive
    let mut symbol = 0;
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
fn nextbit<R: Read>(stream: &mut R, mut cb: u8, mut bits_left: usize)
                                           -> io::Result<(u8, u8, usize)>
{
    if bits_left == 0 {
        cb = try!(stream.read_u8());
        bits_left = 8;

        if cb == 0xff {
            let b2 = try!(stream.read_u8());
            if b2 != 0x0 {
                return error("unexpected marker")
            }
        }
    }

    let r = cb >> 7;
    cb <<= 1;
    bits_left -= 1;
    Ok((r, cb, bits_left))
}

fn reconstruct<R: Read>(dc: &JpegDecoder<R>) -> io::Result<Vec<u8>> {
    let tgt_bytespp = dc.tgt_fmt.bytes_pp();
    let mut result: Vec<u8> = repeat(0).take(dc.w * dc.h * tgt_bytespp).collect();

    match (dc.num_comps, dc.tgt_fmt) {
        (3, ColFmt::RGB) | (3, ColFmt::RGBA) => {
            for ref comp in &dc.comps {
                if comp.sfx != dc.hmax || comp.sfy != dc.vmax {
                    upsample_rgb(dc, &mut result[..]);
                    return Ok(result);
                }
            }

            let mut si = 0;
            let mut di = 0;
            for _j in (0 .. dc.h) {
                for i in (0 .. dc.w) {
                    let pixel = ycbcr_to_rgb(
                        dc.comps[0].data[si+i],
                        dc.comps[1].data[si+i],
                        dc.comps[2].data[si+i],
                    );
                    copy_memory(&pixel[..], &mut result[di..di+3]);
                    if dc.tgt_fmt == ColFmt::RGBA {
                        *result.get_mut(di+3).unwrap() = 255;
                    }
                    di += tgt_bytespp;
                }
                si += dc.num_mcu_x * dc.comps[0].sfx * 8;
            }
            return Ok(result);
        },
        (_, ColFmt::Y) | (_, ColFmt::YA) => {
            let comp = &dc.comps[0];
            if comp.sfx != dc.hmax || comp.sfy != dc.vmax {
                upsample_gray(dc, &mut result[..]);
                return Ok(result);
            }

            // no resampling
            let mut si = 0;
            let mut di = 0;
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
                    copy_memory(&comp.data[si..si+dc.w], &mut result[di..di+dc.w]);
                    si += dc.num_mcu_x * comp.sfx * 8;
                    di += dc.w;
                }
            }
            return Ok(result);
        },
        (1, ColFmt::RGB) | (1, ColFmt::RGBA) => {
            let comp = &dc.comps[0];
            let mut si = 0;
            let mut di = 0;
            for _j in (0 .. dc.h) {
                for i in (0 .. dc.w) {
                    *result.get_mut(di  ).unwrap() = comp.data[si+i];
                    *result.get_mut(di+1).unwrap() = comp.data[si+i];
                    *result.get_mut(di+2).unwrap() = comp.data[si+i];
                    if dc.tgt_fmt == ColFmt::RGBA {
                        *result.get_mut(di+3).unwrap() = 255;
                    }
                    di += tgt_bytespp;
                }
                si += dc.num_mcu_x * comp.sfx * 8;
            }
            return Ok(result);
        },
        _ => return error("internal error"),
    }
}

fn upsample_gray<R: Read>(dc: &JpegDecoder<R>, result: &mut[u8]) {
    let stride0 = dc.num_mcu_x * dc.comps[0].sfx * 8;
    let si0yratio = dc.comps[0].y as f64 / dc.h as f64;
    let si0xratio = dc.comps[0].x as f64 / dc.w as f64;
    let mut di = 0;
    let tgt_bytespp = dc.tgt_fmt.bytes_pp();

    for j in (0 .. dc.h) {
        let si0 = (j as f64 * si0yratio).floor() as usize * stride0;
        for i in (0 .. dc.w) {
            result[di] = dc.comps[0].data[si0 + (i as f64 * si0xratio).floor() as usize];
            if dc.tgt_fmt == ColFmt::YA { result[di+1] = 255; }
            di += tgt_bytespp;
        }
    }
}

fn upsample_rgb<R: Read>(dc: &JpegDecoder<R>, result: &mut[u8]) {
    let stride0 = dc.num_mcu_x * dc.comps[0].sfx * 8;
    let stride1 = dc.num_mcu_x * dc.comps[1].sfx * 8;
    let stride2 = dc.num_mcu_x * dc.comps[2].sfx * 8;
    let si0yratio = dc.comps[0].y as f64 / dc.h as f64;
    let si1yratio = dc.comps[1].y as f64 / dc.h as f64;
    let si2yratio = dc.comps[2].y as f64 / dc.h as f64;
    let si0xratio = dc.comps[0].x as f64 / dc.w as f64;
    let si1xratio = dc.comps[1].x as f64 / dc.w as f64;
    let si2xratio = dc.comps[2].x as f64 / dc.w as f64;

    let mut di = 0;
    let tgt_bytespp = dc.tgt_fmt.bytes_pp();

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
            copy_memory(&pixel[..], &mut result[di..di+3]);
            if dc.tgt_fmt == ColFmt::RGBA { result[di+3] = 255; }
            di += tgt_bytespp;
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
    for i in (0 .. 8) {
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

    let mut i = 0;
    while i < 64 {
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
        i += 8;
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

