My first Rust thing. PNG, TGA, JPEG.
* Only 8-bit images supported
* Returned data is always 8-bit (Y/YA/RGB/RGBA)
* Only decoder for JPEG
* Use `grep -rn 'pub fn' imageformats.rs` to find out more

```Rust
#![feature(globs, macro_rules, slicing_syntax)]

use std::io::{IoResult};
use imageformats::*;
mod imageformats;

fn do_image_io() -> IoResult<()> {
    // last argument defines conversion
    let _pic = try!(read_image("stars.jpg", ColFmt::RGBA));

    // convert to grayscale+alpha
    let _pic = try!(read_image("advanced.png", ColFmt::YA));

    // no conversion
    let pic = try!(read_image("marbles.tga", ColFmt::Auto));

    // write image out as grayscale
    try!(write_image("out.png", pic.w, pic.h, pic.pixels[], ColFmt::Y));

    // print width, heigth and color format (of what you get with ColFmt::Auto)
    println!("{}", read_image_info("hiisi.png"));

    Ok(())
}

fn main() {
    do_image_io().unwrap();
}
```
