// Copyright (c) 2014-2015 Tero Hänninen, license: MIT

use std::io::{Read, Seek, SeekFrom};
use std::mem::{zeroed};
use super::{
    Image, Info, ColFmt, ColType,
    copy_memory, u16_from_be, IFRead,
};

// Baseline JPEG decoder

/// Returns width, height and color type of the image.
pub fn read_info<R: Read+Seek>(reader: &mut R) -> ::Result<Info> {
    let mut marker = [0u8; 2];

    // SOI
    try!(reader.read_exact(&mut marker));
    if &marker[0..2] != &[0xff, 0xd8_u8][..] {
        return Err(::Error::InvalidData("not JPEG"))
    }

    loop {
        try!(reader.read_exact(&mut marker));

        if marker[0] != 0xff { return Err(::Error::InvalidData("no marker")) }
        while marker[1] == 0xff {
            try!(reader.read_exact(&mut marker[1..2]));
        }

        match marker[1] {
            SOF0 | SOF2 => {
                let mut tmp = [0u8; 8];
                try!(reader.read_exact(&mut tmp));
                return Ok(Info {
                    w: u16_from_be(&tmp[5..7]) as usize,
                    h: u16_from_be(&tmp[3..5]) as usize,
                    ct: match tmp[7] {
                           1 => ColType::Gray,
                           3 => ColType::Color,
                           _ => return Err(::Error::InvalidData("not baseline jpeg"))
                       },
                });
            }
            SOS | EOI => return Err(::Error::InvalidData("no frame header")),
            DRI | DHT | DQT | COM | APP0 ... APPF => {
                let mut tmp = [0u8; 2];
                try!(reader.read_exact(&mut tmp));
                let len = u16_from_be(&mut tmp[..]);
                if len < 2 { return Err(::Error::InvalidData("marker length")) }
                try!(reader.seek(SeekFrom::Current(len as i64 - 2)));
            }
            _ => return Err(::Error::InvalidData("invalid / unsupported marker")),
        }
    }
}

pub fn detect<R: Read+Seek>(reader: &mut R) -> bool {
    let start = match reader.seek(SeekFrom::Current(0))
        { Ok(s) => s, Err(_) => return false };
    let result = read_info(reader).is_ok();
    let _ = reader.seek(SeekFrom::Start(start));
    result
}

/// Reads an image and converts it to requested format.
///
/// Passing `ColFmt::Auto` as `req_fmt` converts the data to `Y` or `RGB`.
pub fn read<R: Read+Seek>(reader: &mut R, req_fmt: ColFmt) -> ::Result<Image<u8>> {

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
        correct_comp_ids : true,
        comps       : unsafe { zeroed() },
        num_comps   : 0,
        hmax        : 0,
        vmax        : 0,
    };

    try!(read_markers(dc));   // reads until first scan header

    if dc.eoi_reached {
        return Err(::Error::InvalidData("no image data"))
    }
    dc.tgt_fmt =
        if req_fmt == ColFmt::Auto {
            match dc.num_comps {
                1 => ColFmt::Y,
                3 => ColFmt::RGB,
                _ => return Err(::Error::Internal("wrong format"))
            }
        } else {
            req_fmt
        };

    for comp in dc.comps.iter_mut() {
        comp.data = vec![0u8; dc.num_mcu_x*comp.sfx*8*dc.num_mcu_y*comp.sfy*8];
    }

    Ok(Image::<u8> {
        w   : dc.w,
        h   : dc.h,
        fmt : dc.tgt_fmt,
        buf : {
            // progressive images aren't supported so only one scan
            try!(decode_scan(dc));
            // throw away fill samples and convert to target format
            try!(reconstruct(dc))
        }
    })
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

    correct_comp_ids : bool,
    comps       : [Component; 3],
    num_comps   : usize,
    hmax        : usize,
    vmax        : usize,
}

struct HuffTab {
    values  : [u8; 256],
    mincode : [i16; 16],
    maxcode : [i16; 16],
    valptr  : [i16; 16],
}

struct Component {
    sfx      : usize,            // sampling factor, aka. h
    sfy      : usize,            // sampling factor, aka. v
    x        : usize,          // total number of samples without fill samples
    y        : usize,          // total number of samples without fill samples
    qtable   : usize,
    ac_table : usize,
    dc_table : usize,
    pred     : i16,          // dc prediction
    data     : Vec<u8>,      // reconstructed samples
}

fn read_markers<R: Read+Seek>(dc: &mut JpegDecoder<R>) -> ::Result<()> {
    let mut marker = [0u8; 2];
    // SOI
    try!(dc.stream.read_exact(&mut marker));
    if &marker[0..2] != &[0xff, 0xd8_u8][..] {
        return Err(::Error::InvalidData("not JPEG"))
    }

    let mut has_next_scan_header = false;
    while !has_next_scan_header && !dc.eoi_reached {
        try!(dc.stream.read_exact(&mut marker));

        if marker[0] != 0xff { return Err(::Error::InvalidData("no marker")) }
        while marker[1] == 0xff {
            try!(dc.stream.read_exact(&mut marker[1..2]));
        }

        //println!("marker: 0x{:x}", marker[1]);
        match marker[1] {
            DHT => try!(read_huffman_tables(dc)),
            DQT => try!(read_quantization_tables(dc)),
            SOF0 => {
                if dc.has_frame_header {
                    return Err(::Error::InvalidData("extra frame header"))
                }
                try!(read_frame_header(dc));
                dc.has_frame_header = true;
            }
            SOS => {
                if !dc.has_frame_header {
                    return Err(::Error::InvalidData("no frame header"))
                }
                try!(read_scan_header(dc));
                has_next_scan_header = true;
            }
            DRI => try!(read_restart_interval(dc)),
            EOI => dc.eoi_reached = true,
            APP0 ... APPF | COM => {
                let mut tmp = [0u8; 2];
                try!(dc.stream.read_exact(&mut tmp));
                let len = u16_from_be(&mut tmp[..]);
                if len < 2 { return Err(::Error::InvalidData("invalid data length")) }
                try!(dc.stream.seek(SeekFrom::Current(len as i64 - 2)));
            }
            SOF2 => return Err(::Error::Unsupported("progressive jpeg")),
            _ => return Err(::Error::Unsupported("unsupported marker")),
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
// ...
const APPF: u8 = 0xef;    // application f segment
//const DAC: u8 = 0xcc;     // define arithmetic conditioning table
const COM: u8 = 0xfe;     // comment
const EOI: u8 = 0xd9;     // end of image

fn read_huffman_tables<R: Read>(dc: &mut JpegDecoder<R>) -> ::Result<()> {
    let mut buf = [0u8; 17];
    try!(dc.stream.read_exact(&mut buf[0..2]));
    let mut len = u16_from_be(&buf[0..2]) as isize;
    if len < 2 { return Err(::Error::InvalidData("invalid data length (DHT)")) }
    len -= 2;

    while 0 < len {
        try!(dc.stream.read_exact(&mut buf[0..17]));  // info byte and BITS
        let table_class = buf[0] >> 4;            // 0 = dc table, 1 = ac table
        let table_slot = (buf[0] & 0xf) as usize;  // must be 0 or 1 for baseline
        if 1 < table_slot || 1 < table_class {
            return Err(::Error::InvalidData("invalid table slot or class"))
        }

        // compute total number of huffman codes
        let mut mt = 0;
        for i in 1..17 {
            mt += buf[i] as usize;
        }
        if 256 < mt {
            return Err(::Error::InvalidData("too many huffman codes"));
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

    let mut codes = [0i16; 256];
    let mut sizes = [0u8; 257];

    let mut k = 0;
    for i in 0..16 {
        for _j in 0 .. num_values[i] {
            sizes[k] = (i + 1) as u8;
            k += 1;
        }
    }
    sizes[k] = 0;

    k = 0;
    let mut code = 0_i16;
    let mut si = sizes[k];
    loop {
        while si == sizes[k] {
            codes[k] = code;
            code += 1;
            k += 1;
        }

        if sizes[k] == 0 { break; }

        while si != sizes[k] {
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
    for i in 0..16 {
        mincode[i] = -1;
        maxcode[i] = -1;
        valptr[i] = -1;
    }

    let mut j = 0;
    for i in 0..16 {
        if num_values[i] != 0 {
            valptr[i] = j as i16;
            mincode[i] = codes[j];
            j += (num_values[i] - 1) as usize;
            maxcode[i] = codes[j];
            j += 1;
        }
    }
}

fn read_quantization_tables<R: Read>(dc: &mut JpegDecoder<R>) -> ::Result<()> {
    let mut buf = [0u8; 2];
    try!(dc.stream.read_exact(&mut buf[0..2]));
    let mut len = u16_from_be(&buf[0..2]) as usize;
    if len < 2 { return Err(::Error::InvalidData("invalid data length (DQT)")) }
    len -= 2;
    if len % 65 != 0 {
        return Err(::Error::InvalidData("qtable"))
    }

    while 0 < len {
        try!(dc.stream.read_exact(&mut buf[0..1]));
        let precision = buf[0] >> 4;  // 0 = 8 bit, 1 = 16 bit
        let table_slot = (buf[0] & 0xf) as usize;
        if 3 < table_slot || precision != 0 {   // only 8 bit for baseline
            return Err(::Error::InvalidData("table slot or precision"))
        }
        try!(dc.stream.read_exact(&mut dc.qtables[table_slot][0..64]));
        len -= 65;
    }
    Ok(())
}

fn read_frame_header<R: Read>(dc: &mut JpegDecoder<R>) -> ::Result<()> {
    let mut buf = [0u8; 9];
    try!(dc.stream.read_exact(&mut buf[0..8]));
    let len = u16_from_be(&buf[0..2]) as usize;
    let precision = buf[2];
    dc.h = u16_from_be(&buf[3..5]) as usize;
    dc.w = u16_from_be(&buf[5..7]) as usize;
    dc.num_comps = buf[7] as usize;

    if precision != 8 || (dc.num_comps != 1 && dc.num_comps != 3) ||
       len != 8 + dc.num_comps*3 {
        return Err(::Error::InvalidData("frame header"));
    }

    dc.hmax = 0;
    dc.vmax = 0;
    let mut mcu_du = 0; // data units in one mcu
    try!(dc.stream.read_exact(&mut buf[0 .. dc.num_comps*3]));

    for i in 0 .. dc.num_comps {
        let ci = buf[i*3] as usize;

        // JFIF says ci should be i+1, but there are images where ci is i. Normalize ids
        // so that ci == i, always. So much for standards...
        if i == 0 { dc.correct_comp_ids = ci == i+1; }
        if (dc.correct_comp_ids && ci != i+1)
        || (!dc.correct_comp_ids && ci != i) {
            return Err(::Error::InvalidData("invalid component id"))
        }

        let sampling_factors = buf[i*3 + 1];
        dc.comps[i] = Component {
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
        let comp = &dc.comps[i];
        if comp.sfy < 1 || 4 < comp.sfy ||
           comp.sfx < 1 || 4 < comp.sfx ||
           3 < comp.qtable {
            return Err(::Error::InvalidData("component meta data"))
        }

        if dc.hmax < comp.sfx { dc.hmax = comp.sfx; }
        if dc.vmax < comp.sfy { dc.vmax = comp.sfy; }
        mcu_du += comp.sfx * comp.sfy;
    }
    if 10 < mcu_du { return Err(::Error::InvalidData("du count")) }

    for i in 0 .. dc.num_comps {
        dc.comps[i].x = (dc.w as f64 * dc.comps[i].sfx as f64 / dc.hmax as f64).ceil() as usize;
        dc.comps[i].y = (dc.h as f64 * dc.comps[i].sfy as f64 / dc.vmax as f64).ceil() as usize;
    }

    let mcu_w = dc.hmax * 8;
    let mcu_h = dc.vmax * 8;
    dc.num_mcu_x = (dc.w + mcu_w-1) / mcu_w;
    dc.num_mcu_y = (dc.h + mcu_h-1) / mcu_h;

    Ok(())
}

fn read_scan_header<R: Read>(dc: &mut JpegDecoder<R>) -> ::Result<()> {
    let mut buf = [0u8; 3];
    try!(dc.stream.read_exact(&mut buf));
    let len = u16_from_be(&buf[0..2]) as usize;
    let num_scan_comps = buf[2] as usize;

    if num_scan_comps != dc.num_comps || len != (6+num_scan_comps*2) {
        return Err(::Error::InvalidData("scan header"))
    }

    let mut compbuf = vec![0u8; len-3];
    try!(dc.stream.read_exact(&mut compbuf));

    for i in 0 .. num_scan_comps {
        let ci = (compbuf[i*2] as i32 - if dc.correct_comp_ids { 1 } else { 0 }) as usize;

        if ci >= dc.num_comps {
            return Err(::Error::InvalidData("component id"));
        }

        let tables = compbuf[i*2+1];
        dc.comps[ci].dc_table = (tables >> 4) as usize;
        dc.comps[ci].ac_table = (tables & 0xf) as usize;
        if 1 < dc.comps[ci].dc_table || 1 < dc.comps[ci].ac_table {
            return Err(::Error::InvalidData("dc/ac table index"));
        }
    }

    // ignore last 3 bytes: spectral_start, spectral_end, approx
    Ok(())
}

fn read_restart_interval<R: Read>(dc: &mut JpegDecoder<R>) -> ::Result<()> {
    let mut buf = [0u8; 4];
    try!(dc.stream.read_exact(&mut buf));
    let len = u16_from_be(&buf[0..2]) as usize;
    if len != 4 { return Err(::Error::InvalidData("restart interval")) }
    dc.restart_interval = u16_from_be(&buf[2..4]) as usize;
    Ok(())
}

fn decode_scan<R: Read>(dc: &mut JpegDecoder<R>) -> ::Result<()> {
    let (mut intervals, mut mcus) =
        if 0 < dc.restart_interval {
            let total_mcus = dc.num_mcu_x * dc.num_mcu_y;
            let ivals = (total_mcus + dc.restart_interval-1) / dc.restart_interval;
            (ivals, dc.restart_interval)
        } else {
            (1, dc.num_mcu_x * dc.num_mcu_y)
        };

    let mut block = [0i16; 64];

    for mcu_j in 0 .. dc.num_mcu_y {
        for mcu_i in 0 .. dc.num_mcu_x {

            // decode mcu
            for c in 0 .. dc.num_comps {
                let comp_sfx = dc.comps[c].sfx;
                let comp_sfy = dc.comps[c].sfy;
                let comp_qtab = dc.comps[c].qtable;

                for du_j in 0 .. comp_sfy {
                    for du_i in 0 .. comp_sfx {
                        // decode entropy, dequantize & dezigzag
                        try!(decode_block(dc, c, comp_qtab, &mut block));

                        // idct & level-shift
                        let outx = (mcu_i * comp_sfx + du_i) * 8;
                        let outy = (mcu_j * comp_sfy + du_j) * 8;
                        let dst_stride = dc.num_mcu_x * comp_sfx * 8;
                        let base = &mut dc.comps[c].data[0] as *mut u8;
                        let offset = (outy * dst_stride + outx) as isize;
                        unsafe {
                            let dst = base.offset(offset);
                            stbi_idct_block(dst, dst_stride, &block[..]);
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
                for k in 0 .. dc.num_comps {
                    dc.comps[k].pred = 0;
                }
            }
        }
    }
    Ok(())
}

fn read_restart<R: Read>(stream: &mut R) -> ::Result<()> {
    let mut buf = [0u8; 2];
    try!(stream.read_exact(&mut buf));
    if buf[0] != 0xff || buf[1] < RST0 || RST7 < buf[1] {
        return Err(::Error::InvalidData("reset marker missing"))
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
fn decode_block<R: Read>(dc: &mut JpegDecoder<R>, comp_idx: usize, qtable_idx: usize,
                                                                block: &mut[i16; 64])
                                                                      -> ::Result<()>
{
    let zeros = [0i16; 64];
    unsafe {
        use std::ptr;
        ptr::copy(zeros.as_ptr(), block.as_mut_ptr(), 64);
    }

    let dc_table_idx = dc.comps[comp_idx].dc_table;
    let ac_table_idx = dc.comps[comp_idx].ac_table;
    let t = try!(decode_huff(dc, dc_table_idx, true));
    let diff: i16 = if 0 < t { try!(receive_and_extend(dc, t)) } else { 0 };

    dc.comps[comp_idx].pred += diff;
    block[0] = dc.comps[comp_idx].pred * dc.qtables[qtable_idx][0] as i16;

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
            return Err(::Error::InvalidData("corrupt block"))
        }
        block[DEZIGZAG[k] as usize] =
            try!(receive_and_extend(dc, ssss)) * dc.qtables[qtable_idx][k] as i16;
        k += 1;
    }

    Ok(())
}

fn decode_huff<R: Read>(dc: &mut JpegDecoder<R>, tab_idx: usize, dctab: bool)
                                                            -> ::Result<u8>
{
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
            return Err(::Error::InvalidData("corrupt huffman coding"))
        }
    }
    let j = (tab.valptr[i] - tab.mincode[i] + code) as usize;
    if tab.values.len() <= j {
        return Err(::Error::InvalidData("corrupt huffman coding"))
    }
    Ok(tab.values[j])
}

fn receive_and_extend<R: Read>(dc: &mut JpegDecoder<R>, s: u8) -> ::Result<i16> {
    // receive
    let mut symbol = 0;
    for _ in 0 .. s {
        let (nb, cb, bits_left) = try!(nextbit(dc.stream, dc.cb, dc.bits_left));
        dc.cb = cb;
        dc.bits_left = bits_left;
        symbol = (symbol << 1) + nb as i16;
    }
    // extend
    let vt = 1 << (s - 1);
    if symbol < vt {
        Ok(symbol + (-1 << s) + 1)
    } else {
        Ok(symbol)
    }
}

// returns the bit and the new cb and bits_left
fn nextbit<R: Read>(stream: &mut R, mut cb: u8, mut bits_left: usize)
                                         -> ::Result<(u8, u8, usize)>
{
    if bits_left == 0 {
        cb = try!(stream.read_u8());
        bits_left = 8;

        if cb == 0xff {
            let b2 = try!(stream.read_u8());
            if b2 != 0x0 {
                return Err(::Error::InvalidData("unexpected marker"))
            }
        }
    }

    let r = cb >> 7;
    cb <<= 1;
    bits_left -= 1;
    Ok((r, cb, bits_left))
}

fn reconstruct<R: Read>(dc: &JpegDecoder<R>) -> ::Result<Vec<u8>> {
    let tgt_bytespp = dc.tgt_fmt.channels();
    let mut result = vec![0u8; dc.w * dc.h * tgt_bytespp];

    let (yi, ri, gi, bi, ai) = dc.tgt_fmt.indices_yrgba();

    match (dc.num_comps, dc.tgt_fmt) {
        (3, ColFmt::RGB) | (3, ColFmt::BGR) | (3, ColFmt::RGBA) | (3, ColFmt::BGRA)
                                            | (3, ColFmt::ARGB) | (3, ColFmt::ABGR) => {
            let (sx1, sy1) = (dc.hmax / dc.comps[1].sfx, dc.vmax / dc.comps[1].sfy);
            let (sx2, sy2) = (dc.hmax / dc.comps[2].sfx, dc.vmax / dc.comps[2].sfy);
            let rem = dc.hmax % dc.comps[1].sfx + dc.vmax % dc.comps[1].sfy
                    + dc.hmax % dc.comps[2].sfx + dc.vmax % dc.comps[2].sfy;

            // Use specialized bilinear filtering functions for the frequent cases where
            // Cb & Cr channels have half resolution.
            if dc.comps[0].sfx == dc.hmax && dc.comps[0].sfy == dc.vmax
            && sx1 <= 2 && sx1 <= 2 && sx1 == sx2 && sy1 == sy2 && rem == 0 {
                let resample: fn(&[u8], &[u8], &mut[u8]) =
                    match (sx1, sy1) {
                        (2, 2) => upsample_h2_v2,
                        (2, 1) => upsample_h2_v1,
                        (1, 2) => upsample_h1_v2,
                        (1, 1) => samples_h1_v1,
                        (_, _) => return Err(::Error::Internal("bug")),
                    };

                let mut comp1 = vec![0u8; dc.w];
                let mut comp2 = vec![0u8; dc.w];

                let mut s = 0;
                let mut di = 0;
                for j in 0 .. dc.h {
                    let mi = j / dc.comps[0].sfy;
                    let si = if mi == 0 || mi >= (dc.h-1)/dc.comps[0].sfy { mi }
                                                         else { mi - 1 + s * 2 };
                    s = s ^ 1;

                    let cs = dc.num_mcu_x * dc.comps[1].sfx * 8;
                    let cl0 = mi * cs;
                    let cl1 = si * cs;
                    resample(&dc.comps[1].data[cl0 .. cl0 + dc.comps[1].x],
                             &dc.comps[1].data[cl1 .. cl1 + dc.comps[1].x],
                             &mut comp1[..]);
                    resample(&dc.comps[2].data[cl0 .. cl0 + dc.comps[2].x],
                             &dc.comps[2].data[cl1 .. cl1 + dc.comps[2].x],
                             &mut comp2[..]);

                    for i in 0 .. dc.w {
                        let pixel = ycbcr_to_rgb(
                            dc.comps[0].data[j * dc.num_mcu_x * dc.comps[0].sfx * 8 + i],
                            comp1[i],
                            comp2[i],
                        );
                        result[di+ri] = pixel[0];
                        result[di+gi] = pixel[1];
                        result[di+bi] = pixel[2];
                        if dc.tgt_fmt.has_alpha() == Some(true) { result[di+ai] = 255; }
                        di += tgt_bytespp;
                    }
                }

                return Ok(result)
            }

            // Generic resampling.
            upsample(dc, &mut result[..], ri, gi, bi, ai);
            return Ok(result);
        },
        (_, ColFmt::Y) | (_, ColFmt::YA) | (_, ColFmt::AY) => {
            let comp = &dc.comps[0];
            if comp.sfx != dc.hmax || comp.sfy != dc.vmax {
                upsample_luma(dc, &mut result[..], yi, ai);
                return Ok(result);
            }

            // no resampling
            let mut si = 0;
            let mut di = 0;
            if dc.tgt_fmt.has_alpha() == Some(true) {
                for _j in 0 .. dc.h {
                    for i in 0 .. dc.w {
                        result[di+yi] = comp.data[si+i];
                        result[di+ai] = 255;
                        di += 2;
                    }
                    si += dc.num_mcu_x * comp.sfx * 8;
                }
            } else {    // FmtY
                for _j in 0 .. dc.h {
                    copy_memory(&comp.data[si..si+dc.w], &mut result[di..di+dc.w]);
                    si += dc.num_mcu_x * comp.sfx * 8;
                    di += dc.w;
                }
            }
            return Ok(result);
        },
        (1, ColFmt::RGB) | (1, ColFmt::BGR) | (1, ColFmt::RGBA) | (1, ColFmt::BGRA)
                                            | (1, ColFmt::ARGB) | (1, ColFmt::ABGR) => {
            let comp = &dc.comps[0];
            let mut si = 0;
            let mut di = 0;
            for _j in 0 .. dc.h {
                for i in 0 .. dc.w {
                    result[di+ri] = comp.data[si+i];
                    result[di+gi] = comp.data[si+i];
                    result[di+bi] = comp.data[si+i];
                    if dc.tgt_fmt.has_alpha() == Some(true) {
                        result[di+ai] = 255;
                    }
                    di += tgt_bytespp;
                }
                si += dc.num_mcu_x * comp.sfx * 8;
            }
            return Ok(result);
        },
        _ => return Err(::Error::Internal("error")),
    }
}

fn upsample_h2_v2(line0: &[u8], line1: &[u8], result: &mut[u8]) {
    fn mix(mm: u8, ms: u8, sm: u8, ss: u8) -> u8 {
       (( mm as u32 * 3 * 3
        + ms as u32 * 3 * 1
        + sm as u32 * 1 * 3
        + ss as u32 * 1 * 1
        + 8) / 16) as u8
    }

    result[0] = (( line0[0] as u32 * 3
                 + line1[0] as u32 * 1
                 + 2) / 4) as u8;
    if line0.len() == 1 { return }
    result[1] = mix(line0[0], line0[1], line1[0], line1[1]);

    let mut di = 2;
    for i in 1 .. line0.len() {
        result[di] = mix(line0[i], line0[i-1], line1[i], line1[i-1]);
        di += 1;
        if i == line0.len()-1 {
            if di < result.len() {
                result[di] = (( line0[i] as u32 * 3
                              + line1[i] as u32 * 1
                              + 2) / 4) as u8;
            }
            return;
        }
        result[di] = mix(line0[i], line0[i+1], line1[i], line1[i+1]);
        di += 1;
    }
}

fn upsample_h2_v1(line0: &[u8], _line1: &[u8], result: &mut[u8]) {
    result[0] = line0[0];
    if line0.len() == 1 { return }
    result[1] = (( line0[0] as u32 * 3
                 + line0[1] as u32 * 1
                 + 2) / 4) as u8;
    let mut di = 2;
    for i in 1 .. line0.len() {
        result[di] = (( line0[i-1] as u32 * 1
                      + line0[i+0] as u32 * 3
                      + 2) / 4) as u8;
        di += 1;
        if i == line0.len()-1 {
            if di < result.len() { result[di] = line0[i] };
            return;
        }
        result[di] = (( line0[i+0] as u32 * 3
                      + line0[i+1] as u32 * 1
                      + 2) / 4) as u8;
        di += 1;
    }
}

fn upsample_h1_v2(line0: &[u8], line1: &[u8], result: &mut[u8]) {
    for i in 0 .. result.len() {
        result[i] = (( line0[i] as u32 * 3
                     + line1[i] as u32 * 1
                     + 2) / 4) as u8;
    }
}

fn samples_h1_v1(line0: &[u8], _line1: &[u8], result: &mut[u8]) {
    copy_memory(line0, result)
}

// Nearest neighbor.
fn upsample_luma<R: Read>(dc: &JpegDecoder<R>, result: &mut[u8], li: usize, ai: usize) {
    let stride0 = dc.num_mcu_x * dc.comps[0].sfx * 8;
    let y0_step = dc.comps[0].sfy as f32 / dc.vmax as f32;
    let x0_step = dc.comps[0].sfx as f32 / dc.hmax as f32;

    let mut y0 = y0_step * 0.5;
    let mut y0i = 0;

    let mut di = 0;
    let tgt_bytespp = dc.tgt_fmt.channels();

    for _ in 0 .. dc.h {
        let mut x0 = x0_step * 0.5;
        let mut x0i = 0;
        for _ in 0 .. dc.w {
            result[di+li] = dc.comps[0].data[y0i + x0i];
            if dc.tgt_fmt == ColFmt::YA { result[di+ai] = 255; }
            di += tgt_bytespp;
            x0 += x0_step;
            if x0 >= 1.0 { x0 -= 1.0; x0i += 1; }
        }
        y0 += y0_step;
        if y0 >= 1.0 { y0 -= 1.0; y0i += stride0; }
    }
}

// Generic nearest neighbor
fn upsample<R: Read>(dc: &JpegDecoder<R>, result: &mut[u8], ri: usize, gi: usize,
                                                            bi: usize, ai: usize)
{
    let stride0 = dc.num_mcu_x * dc.comps[0].sfx * 8;
    let stride1 = dc.num_mcu_x * dc.comps[1].sfx * 8;
    let stride2 = dc.num_mcu_x * dc.comps[2].sfx * 8;
    let y0_step = dc.comps[0].sfy as f32 / dc.vmax as f32;
    let y1_step = dc.comps[1].sfy as f32 / dc.vmax as f32;
    let y2_step = dc.comps[2].sfy as f32 / dc.vmax as f32;
    let mut y0 = y0_step * 0.5;
    let mut y1 = y1_step * 0.5;
    let mut y2 = y2_step * 0.5;
    let mut y0i = 0;
    let mut y1i = 0;
    let mut y2i = 0;

    let x0_step = dc.comps[0].sfx as f32 / dc.hmax as f32;
    let x1_step = dc.comps[1].sfx as f32 / dc.hmax as f32;
    let x2_step = dc.comps[2].sfx as f32 / dc.hmax as f32;

    let mut di = 0;
    let tgt_bytespp = dc.tgt_fmt.channels();

    for _j in 0 .. dc.h {
        let mut x0 = x0_step * 0.5;
        let mut x1 = x1_step * 0.5;
        let mut x2 = x2_step * 0.5;
        let mut x0i = 0;
        let mut x1i = 0;
        let mut x2i = 0;
        for _i in 0 .. dc.w {
            let pixel = ycbcr_to_rgb(
                dc.comps[0].data[y0i + x0i],
                dc.comps[1].data[y1i + x1i],
                dc.comps[2].data[y2i + x2i],
            );
            result[di+ri] = pixel[0];
            result[di+gi] = pixel[1];
            result[di+bi] = pixel[2];
            if dc.tgt_fmt.has_alpha() == Some(true) { result[di+ai] = 255; }
            di += tgt_bytespp;
            x0 += x0_step;
            x1 += x1_step;
            x2 += x2_step;
            if x0 >= 1.0 { x0 -= 1.0; x0i += 1; }
            if x1 >= 1.0 { x1 -= 1.0; x1i += 1; }
            if x2 >= 1.0 { x2 -= 1.0; x2i += 1; }
        }
        y0 += y0_step;
        y1 += y1_step;
        y2 += y2_step;
        if y0 >= 1.0 { y0 -= 1.0; y0i += stride0; }
        if y1 >= 1.0 { y1 -= 1.0; y1i += stride1; }
        if y2 >= 1.0 { y2 -= 1.0; y2i += stride2; }
    }
}

#[inline]
fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> [u8; 3] {
    let cb = cb as f32 - 128.0;
    let cr = cr as f32 - 128.0;
    [clamp_to_u8(y as f32 + 1.402*cr),
     clamp_to_u8(y as f32 - 0.34414*cb - 0.71414*cr),
     clamp_to_u8(y as f32 + 1.772*cb)]
}

#[inline]
fn clamp_to_u8(x: f32) -> u8 {
    if x <= 0.0 { return 0; }
    if 255.0 <= x { return 255; }
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
    for i in 0 .. 8 {
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

