# imagefmt  [![Build Status](https://travis-ci.org/lgvz/imagefmt.svg)](https://travis-ci.org/lgvz/imagefmt)

* [Documentation](http://lgvz.github.io/imagefmt/imagefmt/)
* Returned data can be converted to Y, YA, RGB, RGBA, etc.
* All formats are optional via Cargo features (all enabled by default)
* Requires Rust 1.6 or newer

| Format | Decoder                  | Encoder                           |
| ---    | ---                      | ---                               |
| png    | 8-bit, 16-bit            | 8-bit non-paletted non-interlaced |
| tga    | 8-bit non-paletted       | 8-bit non-paletted                |
| bmp    | 8-bit uncompressed       | 8-bit non-paletted uncompressed   |
| jpeg   | baseline non-progressive | nope                              |
