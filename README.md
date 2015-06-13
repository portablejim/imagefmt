**Image loading and saving**
* [Documentation][http://lgvz.github.io/imagefmt/imagefmt/]
* Returned data is always 8-bit (Y/YA/RGB/RGBA/BGR/BGRA)

| Format | Decoder                  | Encoder                           |
| ---    | ---                      | ---                               |
| png    | 8-bit                    | 8-bit non-paletted non-interlaced |
| tga    | 8-bit non-paletted       | 8-bit non-paletted                |
| bmp    | 8-bit uncompressed       | nope                              |
| jpeg   | baseline non-progressive | nope                              |
