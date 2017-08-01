extern crate imagefmt;
use imagefmt::{Image, ColFmt, ColType};

fn equal<T: PartialEq>(pics: &[&Image<T>]) -> bool {
    if pics.len() < 2 { assert!(false) }
    let a = pics[0];
    for &b in &pics[1..] {
        if a.w != b.w || a.h != b.h || a.fmt != b.fmt || a.buf != b.buf {
            return false
        }
    }
    true
}

#[test]
fn png_tga_bmp() {
    // The TGA and BMP files are not as varied in format as the PNG files, so
    // not as well tested.
    let png_path = "tests/pngsuite/";
    let tga_path = "tests/pngsuite-tga/";
    let bmp_path = "tests/pngsuite-bmp/";

    let names = [
        "basi0g08",    // PNG image data, 32 x 32, 8-bit grayscale, interlaced
        "basi2c08",    // PNG image data, 32 x 32, 8-bit/color RGB, interlaced
        "basi3p08",    // PNG image data, 32 x 32, 8-bit colormap, interlaced
        "basi4a08",    // PNG image data, 32 x 32, 8-bit gray+alpha, interlaced
        "basi6a08",    // PNG image data, 32 x 32, 8-bit/color RGBA, interlaced
        "basn0g08",    // PNG image data, 32 x 32, 8-bit grayscale, non-interlaced
        "basn2c08",    // PNG image data, 32 x 32, 8-bit/color RGB, non-interlaced
        "basn3p08",    // PNG image data, 32 x 32, 8-bit colormap, non-interlaced
        "basn4a08",    // PNG image data, 32 x 32, 8-bit gray+alpha, non-interlaced
        "basn6a08",    // PNG image data, 32 x 32, 8-bit/color RGBA, non-interlaced
    ];

    for name in &names {
        let a = imagefmt::read(&format!("{}{}.png", png_path, name), ColFmt::RGBA).unwrap();
        let b = imagefmt::read(&format!("{}{}.tga", tga_path, name), ColFmt::RGBA).unwrap();
        let c = imagefmt::read(&format!("{}{}.bmp", bmp_path, name), ColFmt::RGBA).unwrap();
        assert!(equal(&[&a, &b, &c]));
    }
}

#[test]
fn png_save() {
    let a = imagefmt::read("tests/pngsuite/basn6a08.png", ColFmt::RGBA).unwrap();
    imagefmt::write("tests/temp/testa.png", a.w, a.h, ColFmt::RGBA, &a.buf, ColType::ColorAlpha).unwrap();
    let b = imagefmt::read("tests/temp/testa.png", ColFmt::RGBA).unwrap();
    assert!(equal(&[&a, &b]));
}

#[test]
fn conversions() {
    // Not exhaustive...
    let a = imagefmt::read("tests/pngsuite/basn6a08.png", ColFmt::RGBA).unwrap();
    let b = a.convert(ColFmt::BGRA).unwrap();
    let b = b.convert(ColFmt::ARGB).unwrap();
    let b = b.convert(ColFmt::ABGR).unwrap();
    let b = b.convert(ColFmt::RGBA).unwrap();
    assert!(equal(&[&a, &b]));
    let c = a.convert(ColFmt::RGB).unwrap();
    let d = a.convert(ColFmt::BGR).unwrap();
    let e = d.convert(ColFmt::RGB).unwrap();
    assert!(equal(&[&c, &e]));
    let ay = a.convert(ColFmt::AY).unwrap();
    let ya = b.convert(ColFmt::YA).unwrap();
    let b = ya.convert(ColFmt::AY).unwrap();
    assert!(equal(&[&ay, &b]));
    let b = b.convert(ColFmt::Y).unwrap();
    let c = c.convert(ColFmt::Y).unwrap();
    let d = d.convert(ColFmt::Y).unwrap();
    let g = a.convert(ColFmt::Y).unwrap();
    assert!(equal(&[&b, &c, &d, &g]));

    // From gray to color+alpha.
    let rgba = g.convert(ColFmt::RGBA).unwrap();
    for &fmt in &[ColFmt::BGRA, ColFmt::ARGB, ColFmt::ABGR] {
        let ca = g.convert(fmt).unwrap();
        let ca = ca.convert(ColFmt::RGBA).unwrap();
        assert!(equal(&[&ca, &rgba]));
    }

    // From gray+alpha to color+alpha.
    let rgba0 = ya.convert(ColFmt::RGBA).unwrap();
    let rgba1 = ay.convert(ColFmt::RGBA).unwrap();
    for &fmt in &[ColFmt::BGRA, ColFmt::ARGB, ColFmt::ABGR] {
        let ca0 = ya.convert(fmt).unwrap();
        let ca0 = ca0.convert(ColFmt::RGBA).unwrap();
        let ca1 = ay.convert(fmt).unwrap();
        let ca1 = ca1.convert(ColFmt::RGBA).unwrap();
        assert!(equal(&[&ca0, &ca1, &rgba0, &rgba1]));
    }
}
