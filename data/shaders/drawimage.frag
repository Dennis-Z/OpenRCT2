#version 150

const int FLAG_COLOUR              = (1 << 0);
const int FLAG_REMAP               = (1 << 1);
const int FLAG_TRANSPARENT         = (1 << 2);
const int FLAG_TRANSPARENT_SPECIAL = (1 << 3);

uniform vec4            uPalette[256];
uniform usampler2DArray uTexture;
uniform sampler3D       uLightmap;

flat in ivec4           fClip;
flat in int             fFlags;
in vec4                 fColour;
flat in int             fTexColourAtlas;
in vec2                 fTexColourCoords;
flat in int             fTexMaskAtlas;
in vec2                 fTexMaskCoords;
flat in int             fTexPaletteAtlas;
flat in vec4            fTexPaletteBounds;
flat in int             fMask;
flat in vec3            fWorldBoxOrigin;
flat in vec3            fBoxProjData;
flat in float           fPrelight;

in vec2 fPosition;
in vec2 fTextureCoordinate;

out vec4 oColour;

void main()
{
    if (fPosition.x < fClip.x || fPosition.x > fClip.z ||
        fPosition.y < fClip.y || fPosition.y > fClip.w)
    {
        discard;
    }

    vec4 texel;

    // If remap palette used
    if ((fFlags & FLAG_REMAP) != 0)
    {
        // z is the size of each x pixel in the atlas
        float x = fTexPaletteBounds.x + texture(uTexture, vec3(fTexColourCoords, float(fTexColourAtlas))).r * fTexPaletteBounds.z;
        texel = uPalette[texture(uTexture, vec3(x, fTexPaletteBounds.y, float(fTexPaletteAtlas))).r];
    } // If transparent or special transparent
    else if ((fFlags & (FLAG_TRANSPARENT | FLAG_TRANSPARENT_SPECIAL)) != 0)
    {
        float line = texture(uTexture,vec3(fTexColourCoords, float(fTexColourAtlas))).r;
        if (line == 0.0)
        {
            discard;
        }
        float alpha = 0.5;
        if ((fFlags & FLAG_TRANSPARENT_SPECIAL) != 0)
        {
            alpha = 0.5 + (line - 1.0) / 10.0;
        }
    
        // z is the size of each x pixel in the atlas
        float x = fTexPaletteBounds.x + fTexPaletteBounds.z * 50.0;
        oColour = vec4(uPalette[texture(uTexture, vec3(x, fTexPaletteBounds.y, float(fTexPaletteAtlas))).r].rgb, alpha);
        
        return;
    }
    else
    {
        texel = uPalette[texture(uTexture, vec3(fTexColourCoords, float(fTexColourAtlas))).r];
    }

    if (fMask != 0)
    {
        float mask = texture(uTexture, vec3(fTexMaskCoords, float(fTexMaskAtlas))).r;
        if ( mask == 0.0 )
        {
            discard;
        }

        oColour = texel;
    }
    else
    {
        if ((fFlags & FLAG_COLOUR) != 0)
        {
            oColour = vec4(fColour.rgb, fColour.a * texel.a);
        }
        else
        {
            if (fPrelight > 0.5) {
                oColour = texel;
            } else {
                float diffFromXSplit = fBoxProjData.x - fPosition.x;
                float height = (fBoxProjData.y - fPosition.y) - abs(diffFromXSplit) * 0.5;
                bool isFlat = height > fBoxProjData.y - fBoxProjData.z;
                vec3 worldPos = fWorldBoxOrigin;
                if (isFlat) {
                    float extraY = height - (fBoxProjData.y - fBoxProjData.z);
                    worldPos.x += extraY / 32;
                    worldPos.y += extraY / 32;
                    worldPos.z += (fBoxProjData.y - fBoxProjData.z) / 8;
                    if (diffFromXSplit > 0) {
                        worldPos.y += diffFromXSplit / 32;
                    } else {
                        worldPos.x -= diffFromXSplit / 32;
                    }
                } else {
                    oColour = vec4(height * 0.01, 0, 0, texel.a);
                    worldPos.z += height / 8;
                    if (diffFromXSplit > 0) {
                        worldPos.y += diffFromXSplit / 32;
                    } else {
                        worldPos.x -= diffFromXSplit / 32;
                    }

                }
                
                //oColour = vec4(worldPos.xy * 0.01, 0, texel.a);
                vec3 lmPos = worldPos * vec3(2.0, 2.0, 1.0);
                vec3 sampleVec = lmPos / vec3(512.0, 512.0, 64.0);
                float lmint = texture(uLightmap, sampleVec).r;
                //oColour = vec4(texture(uLightmap, sampleVec).r, texel.g * 0.2, texel.b * 0.2, texel.a);
                oColour = texel;
                oColour.rgb *= lmint;

                //oColour = vec4(mod(worldPos.xyz * 0.1, 1), texel.a);
            }
            //oColour = vec4(fWorldBoxOrigin.rgb * 0.01, texel.a + 0.5);
            //oColour = texel;
            //oColour = vec4(fPosition * 0.01, 0, 1);
        }
    }
}
