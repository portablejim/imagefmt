extern crate imagefmt;
use imagefmt::ColFmt;

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
        assert_eq!(a.w, b.w); assert_eq!(a.w, c.w);
        assert_eq!(a.h, b.h); assert_eq!(a.h, c.h);
        assert_eq!(a.buf.len(), b.buf.len()); assert_eq!(a.buf.len(), c.buf.len());
        assert!(a.buf == b.buf); assert!(a.buf == c.buf);
    }
}
