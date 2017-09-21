#version 150

const int FLAG_COLOUR              = (1 << 0);
const int FLAG_REMAP               = (1 << 1);
const int FLAG_TRANSPARENT         = (1 << 2);
const int FLAG_TRANSPARENT_SPECIAL = (1 << 3);

uniform vec4            uPalette[256];
uniform usampler2DArray uTexture;
uniform usampler2DArray uDisplacementTexture;
uniform sampler3D       uLightmap;
uniform sampler3D       uLightmapInterpolate;
uniform mat2            uRotationTransform;

flat in ivec4           fClip;
flat in int             fFlags;
in vec4                 fColour;
flat in int             fTexColourAtlas;
in vec2                 fTexColourCoords;
flat in int             fTexDisplacementAtlas;
in vec2                 fTexDisplacementCoords;
flat in int             fTexMaskAtlas;
in vec2                 fTexMaskCoords;
flat in int             fTexPaletteAtlas;
flat in vec4            fTexPaletteBounds;
flat in int             fMask;
flat in vec4            fWorldIn;

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

    // Lighting
    vec3 worldPos = fWorldIn.xyz;
    float prelight = fWorldIn.w;

    // reads uvec!
    vec3 worldOffset = vec3(texture(uDisplacementTexture, vec3(fTexDisplacementCoords, float(fTexDisplacementAtlas))).xyz) / vec3(64.0, 64.0, 8.0);
    worldOffset += vec3(-0.075, -0.075, 0.075); // small offset in sprite displacement space, this kicks the faces in the towards-screen cell
    worldOffset.xy = uRotationTransform * worldOffset.xy;
    worldPos += worldOffset;
    
    vec3 lmPos = worldPos * vec3(2.0, 2.0, 1.0);
    vec3 lmSamplePos = lmPos - vec3(0.5);
    vec3 lmSize = vec3(512.0, 512.0, 128.0);

    // Lightmap
    float sampleInterpX = texture(uLightmapInterpolate, vec3(round(lmPos.x) + 0.5, lmPos.y, lmPos.z) / lmSize).x;
    float sampleInterpY = texture(uLightmapInterpolate, vec3(lmPos.x, round(lmPos.y) + 0.5, lmPos.z) / lmSize).y;
    float sampleInterpZ = texture(uLightmapInterpolate, vec3(lmPos.x, lmPos.y, round(lmPos.z) + 0.5) / lmSize).z;
    float fracX = sampleInterpX < 0.5 ? (fract(lmSamplePos.x) > 0.5 ? 1.0 : 0.0) : fract(lmSamplePos.x);
    float fracY = sampleInterpY < 0.5 ? (fract(lmSamplePos.y) > 0.5 ? 1.0 : 0.0) : fract(lmSamplePos.y);
    float fracZ = sampleInterpZ < 0.5 ? (fract(lmSamplePos.z) > 0.5 ? 1.0 : 0.0) : fract(lmSamplePos.z);
    vec3 sample000 = texture(uLightmap, (floor(lmSamplePos + vec3(0.0, 0.0, 0.0)) + vec3(0.5)) / lmSize).rgb;
    vec3 sample100 = texture(uLightmap, (floor(lmSamplePos + vec3(1.0, 0.0, 0.0)) + vec3(0.5)) / lmSize).rgb;
    vec3 sample010 = texture(uLightmap, (floor(lmSamplePos + vec3(0.0, 1.0, 0.0)) + vec3(0.5)) / lmSize).rgb;
    vec3 sample110 = texture(uLightmap, (floor(lmSamplePos + vec3(1.0, 1.0, 0.0)) + vec3(0.5)) / lmSize).rgb;
    vec3 sample001 = texture(uLightmap, (floor(lmSamplePos + vec3(0.0, 0.0, 1.0)) + vec3(0.5)) / lmSize).rgb;
    vec3 sample101 = texture(uLightmap, (floor(lmSamplePos + vec3(1.0, 0.0, 1.0)) + vec3(0.5)) / lmSize).rgb;
    vec3 sample011 = texture(uLightmap, (floor(lmSamplePos + vec3(0.0, 1.0, 1.0)) + vec3(0.5)) / lmSize).rgb;
    vec3 sample111 = texture(uLightmap, (floor(lmSamplePos + vec3(1.0, 1.0, 1.0)) + vec3(0.5)) / lmSize).rgb;
    vec3 sample00 = mix(sample000, sample001, fracZ);
    vec3 sample10 = mix(sample100, sample101, fracZ);
    vec3 sample01 = mix(sample010, sample011, fracZ);
    vec3 sample11 = mix(sample110, sample111, fracZ);
    vec3 sample0 = mix(sample00, sample01, fracY);
    vec3 sample1 = mix(sample10, sample11, fracY);
    vec3 sample = mix(sample0, sample1, fracX);
    vec3 lightValue = sample;

    // AO
    float sampleInterpAOX = min(1.0, 2*texture(uLightmapInterpolate, vec3(lmPos.x + 0.5, lmPos.y, lmPos.z) / lmSize).x);
    float sampleInterpAOY = min(1.0, 2*texture(uLightmapInterpolate, vec3(lmPos.x, lmPos.y + 0.5, lmPos.z) / lmSize).y);
    float sampleInterpAOZ = min(1.0, 4*texture(uLightmapInterpolate, vec3(lmPos.x, lmPos.y, lmPos.z + 0.5) / lmSize).z);

    float aoXv = length(vec3(sampleInterpAOX, sampleInterpAOY, sampleInterpAOZ));
    float aoIntensity = pow(min(1.0, aoXv / sqrt(2)), 0.5);
    lightValue *= aoIntensity;

    vec4 lmmultint = mix(vec4(lightValue * 1.5 + 0.01, 1), vec4(1, 1, 1, 1), prelight);
    vec4 lmaddint = mix(vec4(max(lightValue - 0.25, 0) * 0.5, 0), vec4(0, 0, 0, 0), prelight) * 0;

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
        oColour = vec4(pow(uPalette[texture(uTexture, vec3(x, fTexPaletteBounds.y, float(fTexPaletteAtlas))).r].rgb, vec3(2.2)), alpha) * lmmultint + lmaddint;
        oColour.rgb = pow(oColour.rgb, vec3(1.0 / 2.2));
        
        return;
    }
    else
    {
        texel = uPalette[texture(uTexture, vec3(fTexColourCoords, float(fTexColourAtlas))).r];
    }

    texel.rgb = pow(texel.rgb, vec3(2.2));

    if (fMask != 0)
    {
        float mask = texture(uTexture, vec3(fTexMaskCoords, float(fTexMaskAtlas))).r;
        if ( mask == 0.0 )
        {
            discard;
        }

        oColour = texel * lmmultint + lmaddint;
    }
    else
    {
        if ((fFlags & FLAG_COLOUR) != 0)
        {
            oColour = vec4(fColour.rgb, fColour.a * texel.a) * lmmultint + lmaddint;
        }
        else
        {
            oColour = texel * lmmultint + lmaddint;
        }
    }

    oColour.rgb = pow(oColour.rgb, vec3(1.0 / 2.2));
}
