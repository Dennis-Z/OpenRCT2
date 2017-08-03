#version 150

uniform ivec2 uScreenSize;

in ivec4 ivClip;
in int   ivTexColourAtlas;
in vec4  ivTexColourBounds;
in int   ivTexMaskAtlas;
in vec4  ivTexMaskBounds;
in int   ivTexPaletteAtlas;
in vec4  ivTexPaletteBounds;
in int   ivFlags;
in vec4  ivColour;
in ivec4 ivBounds;
in int   ivMask;
in vec3  ivWorldBoxOrigin;
in vec3  ivWorldBoxSize;
in float ivPrelight;

in uint vIndex;

out vec2       fTextureCoordinate;
out vec2       fPosition;
flat out ivec4 fClip;
flat out int   fFlags;
out vec4       fColour;
flat out int   fTexColourAtlas;
out vec2       fTexColourCoords;
flat out int   fTexMaskAtlas;
out vec2       fTexMaskCoords;
flat out int   fTexPaletteAtlas;
flat out vec4  fTexPaletteBounds;
flat out int   fMask;
flat out vec3  fWorldBoxOrigin;
flat out vec3  fBoxProjData;
flat out float fPrelight;

void main()
{
    vec2 pos;
    switch (vIndex) {
    case 0u:
        pos = ivBounds.xy;
        fTexColourCoords = ivTexColourBounds.xy;
        fTexMaskCoords = ivTexMaskBounds.xy;
        break;
    case 1u:
        pos = ivBounds.zy;
        fTexColourCoords = ivTexColourBounds.zy;
        fTexMaskCoords = ivTexMaskBounds.zy;
        break;
    case 2u:
        pos = ivBounds.xw;
        fTexColourCoords = ivTexColourBounds.xw;
        fTexMaskCoords = ivTexMaskBounds.xw;
        break;
    case 3u:
        pos = ivBounds.zw;
        fTexColourCoords = ivTexColourBounds.zw;
        fTexMaskCoords = ivTexMaskBounds.zw;
        break;
    }

    fPosition = pos;

    // Transform screen coordinates to viewport
    pos.x = (pos.x * (2.0 / uScreenSize.x)) - 1.0;
    pos.y = (pos.y * (2.0 / uScreenSize.y)) - 1.0;
    pos.y *= -1;

    fClip = ivClip;
    fFlags = ivFlags;
    fColour = ivColour;
    fMask = ivMask;
    fTexColourAtlas = ivTexColourAtlas;
    fTexMaskAtlas = ivTexMaskAtlas;
    fTexPaletteAtlas = ivTexPaletteAtlas;
    fTexPaletteBounds = ivTexPaletteBounds;
    fWorldBoxOrigin = ivWorldBoxOrigin;
    fBoxProjData = ivWorldBoxSize;
    fPrelight = ivPrelight;

    gl_Position = vec4(pos, 0.0, 1.0);
}
