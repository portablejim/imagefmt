My first Rust thing. PNG, TGA, JPEG.
* Only 8-bit images supported
* Returned data is always 8-bit (Y/YA/RGB/RGBA)
* Only decoder for JPEG
* Use `grep -rn 'pub fn' imageformats.rs` to find out more

```Rust
#![feature(globs)]
#![feature(macro_rules)]

use std::io::{IoResult};
use imageformats::*;
mod imageformats;

fn do_image_io() -> IoResult<()> {
    // last argument defines conversion
    let _pic = try!(read_image("stars.jpg", FmtRGBA));

    // convert to grayscale+alpha
    let _pic = try!(read_image("advanced.png", FmtYA));

    // no conversion
    let pic = try!(read_image("marbles.tga", FmtAuto));

    // write image out as grayscale
    try!(write_image("out.png", pic.w, pic.h, pic.pixels[], FmtY));

    // print width, heigth and color format (of what you get with FmtAuto)
    println!("{}", read_image_info("hiisi.png"));

    Ok(())
}

fn main() {
    match do_image_io() {
        Ok(r) => r,
        Err(e) => fail!("failed: {}", e)
    };
}
```
