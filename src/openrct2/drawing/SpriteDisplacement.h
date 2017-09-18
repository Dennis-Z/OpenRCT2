#define SPRITESHAPE_SIZE_X 64
#define SPRITESHAPE_SIZE_Y 64

#include "../common.h"

#ifdef __cplusplus

#include <bitset>
#include <vector>

typedef std::bitset<SPRITESHAPE_SIZE_X * SPRITESHAPE_SIZE_Y> SpriteShape;

class SpriteDisplacement {
public:
    SpriteDisplacement(uint16 width, uint16 height) : width(width), height(height), data(width * height * 3) {}
    uint16 width;
    uint16 height;
    std::vector<uint8> data;
};

SpriteShape get_sprite_shape(uint32 image);
SpriteDisplacement get_sprite_displacement(uint32 image);

extern "C" {
#endif

int load_known_displacements();
int link_shape_to_displacement_file(uint32 image, const char* fname);

#ifdef __cplusplus
}
#endif