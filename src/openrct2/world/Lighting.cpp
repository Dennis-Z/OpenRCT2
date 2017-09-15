extern "C" {
    #include "lighting.h"
    #include "../world/map.h"
    #include "../world/scenery.h"
    #include "../world/footpath.h"
}

#include <atomic>
#include <mutex>
#include <thread>
#include <queue>
#include <unordered_set>
#include "../core/Math.hpp"

// how the light will be affected when light passes through a certain plane
lighting_value* lightingAffectorsX = NULL;
lighting_value* lightingAffectorsY = NULL;
lighting_value* lightingAffectorsZ = NULL;
#define LAIDX(y, x, z) ((y) * (LIGHTMAP_SIZE_X+1) * (LIGHTMAP_SIZE_Z+1) + (x) * (LIGHTMAP_SIZE_Z+1) + (z))

// lightmap columns whose lightingAffectors are outdated
// each element contains a bitmap, each bit identifying if a direction should be recomputed (4 bits, 1 << MAP_ELEMENT_DIRECTION_... for each direction)
// Z is always recomputed
uint8 affectorRecomputeQueue[LIGHTMAP_SIZE_Y][LIGHTMAP_SIZE_X];

//lighting_chunk lightingChunks[LIGHTMAP_CHUNKS_Z][LIGHTMAP_CHUNKS_Y][LIGHTMAP_CHUNKS_X];

lighting_chunk* lightingChunks = NULL;
#define LIGHTINGCHUNK(z, y, x) lightingChunks[(z) * (LIGHTMAP_CHUNKS_Y) * (LIGHTMAP_CHUNKS_X) + (y) * (LIGHTMAP_CHUNKS_Y) + (x)]

#define LIGHTMAXSPREAD 7

const lighting_value black = { 0, 0, 0 };
const lighting_value dimmedblack = { 200, 200, 200 };
const lighting_value dimmedblackside = { 200, 200, 200 };
const lighting_value dimmedblackvside = { 250, 250, 250 };
//const lighting_value ambient_sky = { 10 / 2, 20 / 2, 70 / 2 };
const lighting_value ambient_sky = { 150, 130, 120 };
const lighting_value lit = { 255, 255, 255 };
const lighting_value lightlit = { 180/2, 110/2, 108/2 };

#define SUBCELLITR(v, cbidx) for (int v = (cbidx); v < (cbidx) + 2; v++)
#define CHUNKRANGEITRXY(lm_x, lm_y, sx, sy, range)              for (int sy = Math::Max(0, ((lm_y) - (range)) / LIGHTMAP_CHUNK_SIZE); sy <= Math::Min(LIGHTMAP_CHUNKS_Y - 1, ((lm_y) + (range)) / LIGHTMAP_CHUNK_SIZE); sy++)\
                                                                    for (int sx = Math::Max(0, ((lm_x) - (range)) / LIGHTMAP_CHUNK_SIZE); sx <= Math::Min(LIGHTMAP_CHUNKS_X - 1, (lm_x + (range)) / LIGHTMAP_CHUNK_SIZE); sx++)
#define CHUNKRANGEITRXYZ(lm_x, lm_y, lm_z, sx, sy, sz, range)   for (int sz = Math::Max(0, ((lm_z) - (range) * 2) / LIGHTMAP_CHUNK_SIZE); sz <= Math::Min(LIGHTMAP_CHUNKS_Z - 1, ((lm_z) + (range) * 2) / LIGHTMAP_CHUNK_SIZE); sz++)\
                                                                    CHUNKRANGEITRXY((lm_x), (lm_y), sx, sy, (range))

template <typename T> class queue_set {
private:
    std::queue<T> queue;
    std::unordered_set<T> set;

public:
    bool empty() { return queue.empty(); }

    void clear() {
        std::queue<T>().swap(queue);
        std::unordered_set<T>().swap(set);
    }

    T frontpop() {
        T front = queue.front();
        queue.pop();
        set.erase(front);
        return front;
    }
    
    void push(T elem) {
        if (set.find(elem) == set.end()) {
            set.insert(elem);
            queue.push(elem);
        }
    }
};

// TODO: custom data structure for queue + set
queue_set<lighting_chunk*> outdated_skylight; // should recompute skylight
std::mutex outdated_skylight_mutex;
std::mutex outdated_gpu_mutex;
queue_set<lighting_chunk*> outdated_static; // has outdated static lights
queue_set<lighting_chunk*> outdated_gpu; // needs to update texture on gpu
std::unordered_set<lighting_chunk*> dynamic_chunks;

float skylight_direction[3] = { 0.0, 0.0, 0.0 }; // normalized with manhattan distance(!) x + y + z = 1
rct_xyz16 skylight_delta = {0, 0, 0}; // each value +1 or -1, depending on the direction the skylight travels
std::vector<lighting_chunk*> skylight_batch[LIGHTMAP_CHUNKS_X + LIGHTMAP_CHUNKS_Y + LIGHTMAP_CHUNKS_Z]; // indexed by distance to corner
int skylight_batch_current = 0;
std::atomic<uint8> skylight_batch_remaining;
rct_xyz16 skylight_cell_itr[LIGHTMAP_CHUNK_SIZE * LIGHTMAP_CHUNK_SIZE * LIGHTMAP_CHUNK_SIZE];

std::mutex is_collecting_data_mutex;
bool is_collecting_data = false;
std::atomic<bool> worker_threads_continue;
std::vector<std::thread> worker_threads;

// multiplies @target light with some multiplier light value @apply
static void lighting_multiply(lighting_value* target, const lighting_value apply) {
    // don't convert to FP
    uint16 mulr = ((uint16)target->r * apply.r) / 255;
    uint16 mulg = ((uint16)target->g * apply.g) / 255;
    uint16 mulb = ((uint16)target->b * apply.b) / 255;
    target->r = mulr;
    target->g = mulg;
    target->b = mulb;
}

// adds @target light with some multiplier light value @apply
static void lighting_add(lighting_value* target, const lighting_value apply) {
    uint16 resr = (uint16)target->r + apply.r;
    uint16 resg = (uint16)target->g + apply.g;
    uint16 resb = (uint16)target->b + apply.b;
    target->r = resr > 255 ? 255 : resr;
    target->g = resg > 255 ? 255 : resg;
    target->b = resb > 255 ? 255 : resb;
}

/*
// elementswise lerp between @a and @b depending on @frac (@lerp = 0 -> @a)
// @return 
static lighting_value interpolate_lighting(const lighting_value a, const lighting_value b, float frac) {
    return {
        (uint8)(a.r * (1.0f - frac) + b.r * frac),
        (uint8)(a.g * (1.0f - frac) + b.g * frac),
        (uint8)(a.b * (1.0f - frac) + b.b * frac)
    };
}
*/

static float max_intensity_at(lighting_light light, int chlm_x, int chlm_y, int chlm_z) {
    sint32 w_x = chlm_x * 16 + 8;
    sint32 w_y = chlm_y * 16 + 8;
    sint32 w_z = chlm_z * 2 + 1;
    float distpot = ((w_x - light.pos.x)*(w_x - light.pos.x) + (w_y - light.pos.y)*(w_y - light.pos.y) + (w_z - light.pos.z)*(w_z - light.pos.z) * 4 * 4);
    float intensity = 0.5f;
    if (distpot != 0.f)
    {
        intensity = 500.0f / (distpot);
    }
    if (intensity > 0.5f) intensity = 0.5f;
    return intensity;
}

// expands a light into the map array passed
static void light_expand_to_map(lighting_light light, lighting_value map[LIGHTMAXSPREAD * 4 + 1][LIGHTMAXSPREAD * 2 + 1][LIGHTMAXSPREAD * 2 + 1]) {
    float x = light.pos.x / (32.0f / LIGHTING_CELL_SUBDIVISIONS);
    float y = light.pos.y / (32.0f / LIGHTING_CELL_SUBDIVISIONS);
    float z = light.pos.z / 2.0f;
    int lm_x = x;
    int lm_y = y;
    int lm_z = z;

    // light offset from the center of the cell in [-0.5, 0.5]
    float off_x = (x - lm_x) - 0.5f;
    float off_y = (y - lm_y) - 0.5f;
    float off_z = (z - lm_z) - 0.5f;

    float intensity000 = max_intensity_at(light, lm_x, lm_y, lm_z); // to the "root" light cell
    map[LIGHTMAXSPREAD * 2][LIGHTMAXSPREAD][LIGHTMAXSPREAD] = lightlit;
    // apply falloff at the center cell (light may not be perfectly rounded to the cell center)
    // note that this falloff can be "recovered" when spreading to nearby cells below if the light source is e.g. on the edge of two lightmap cells
    map[LIGHTMAXSPREAD * 2][LIGHTMAXSPREAD][LIGHTMAXSPREAD].r *= intensity000;
    map[LIGHTMAXSPREAD * 2][LIGHTMAXSPREAD][LIGHTMAXSPREAD].g *= intensity000;
    map[LIGHTMAXSPREAD * 2][LIGHTMAXSPREAD][LIGHTMAXSPREAD].b *= intensity000;

    // temporary cache
    static rct_xyz16 itr_queue[(LIGHTMAXSPREAD * 4 + 1) * (LIGHTMAXSPREAD * 2 + 1) * (LIGHTMAXSPREAD * 2 + 1)];
    static bool did_compute_itr_queue = false;

    if (!did_compute_itr_queue) {
        int itr_queue_build_pos = 0;
        for (sint16 dist = 0; dist <= LIGHTMAXSPREAD + LIGHTMAXSPREAD + LIGHTMAXSPREAD * 2; dist++) {
            for (sint16 dx = -LIGHTMAXSPREAD; dx <= LIGHTMAXSPREAD; dx++)
                for (sint16 dy = -LIGHTMAXSPREAD; dy <= LIGHTMAXSPREAD; dy++)
                    for (sint16 dz = -LIGHTMAXSPREAD * 2; dz <= LIGHTMAXSPREAD * 2; dz++) {
                        int thisdist = abs(dx) + abs(dy) + abs(dz);
                        if (thisdist == dist) itr_queue[itr_queue_build_pos++] = { dx, dy, dz };
                    }
        }

        did_compute_itr_queue = true;
    }

    // iterate distances (skip 0, 0, 0)
    for (int citr = 1; citr < (LIGHTMAXSPREAD * 4 + 1) * (LIGHTMAXSPREAD * 2 + 1) * (LIGHTMAXSPREAD * 2 + 1); citr++) {
        rct_xyz16* delta = itr_queue + citr;
        // distance to the light in lightmap space
        float this_delta_x = delta->x - off_x;
        float this_delta_y = delta->y - off_y;
        float this_delta_z = (delta->z - off_z) * 0.5;
        // this forces not reading values that have not yet been set (i.e. going outwards from the center)
        if (delta->x == 0) this_delta_x = 0.0f;
        if (delta->y == 0) this_delta_y = 0.0f;
        if (delta->z == 0) this_delta_z = 0.0f;

        int w_x = lm_x + delta->x;
        int w_y = lm_y + delta->y;
        int w_z = lm_z + delta->z;
        // manhattan distance, ensures that fragx/y/z below sum to 1
        float dist = fabs(this_delta_x) + fabs(this_delta_y) + fabs(this_delta_z);
        // delta/dist -> how much from each direction should be consumed? (sums to 1 across all axes)
        // intensitybase / intensity -> how much should the light intensity fall off when making this jump (always in [0.0, 1.0])?
        float intensitybase = max_intensity_at(light, lm_x + delta->x, lm_y + delta->y, lm_z + delta->z);
        float fragx = fabs(this_delta_x) / dist * intensitybase / max_intensity_at(light, w_x + (delta->x < 0 ? 1 : -1), w_y, w_z);
        float fragy = fabs(this_delta_y) / dist * intensitybase / max_intensity_at(light, w_x, w_y + (delta->y < 0 ? 1 : -1), w_z);
        float fragz = fabs(this_delta_z) / dist * intensitybase / max_intensity_at(light, w_x, w_y, w_z + (delta->z < 0 ? 1 : -1));

        // lighting data from the source positions
        lighting_value from_x = map[LIGHTMAXSPREAD * 2 + delta->z][LIGHTMAXSPREAD + delta->y][LIGHTMAXSPREAD + delta->x + (delta->x < 0 ? 1 : -1)];
        lighting_value from_y = map[LIGHTMAXSPREAD * 2 + delta->z][LIGHTMAXSPREAD + delta->y + (delta->y < 0 ? 1 : -1)][LIGHTMAXSPREAD + delta->x];
        lighting_value from_z = map[LIGHTMAXSPREAD * 2 + delta->z + (delta->z < 0 ? 1 : -1)][LIGHTMAXSPREAD + delta->y][LIGHTMAXSPREAD + delta->x];

        // apply affectors from the boundaries
        // TODO: maybe quad-lerp like done with raycasts? will yield smoother occlusion, especially when a light is moving
        //       will impact performance though (and still not look as smooth as the raycast approach)
        lighting_multiply(&from_x, lightingAffectorsX[LAIDX(w_y, w_x + (delta->x < 0), w_z)]);
        lighting_multiply(&from_y, lightingAffectorsY[LAIDX(w_y + (delta->y < 0), w_x, w_z)]);
        lighting_multiply(&from_z, lightingAffectorsZ[LAIDX(w_y, w_x, w_z + (delta->z < 0))]);

        // interpolate values
        map[LIGHTMAXSPREAD * 2 + delta->z][LIGHTMAXSPREAD + delta->y][LIGHTMAXSPREAD + delta->x] = {
            (uint8)(from_x.r * fragx + from_y.r * fragy + from_z.r * fragz),
            (uint8)(from_x.g * fragx + from_y.g * fragy + from_z.g * fragz),
            (uint8)(from_x.b * fragx + from_y.b * fragy + from_z.b * fragz)
        };
    }
}

// given an expanded light map, applies it to a chunk
static void light_expansion_apply(lighting_light light, lighting_value map[LIGHTMAXSPREAD * 4 + 1][LIGHTMAXSPREAD * 2 + 1][LIGHTMAXSPREAD * 2 + 1], lighting_chunk* target, lighting_value target_data[LIGHTMAP_CHUNK_SIZE][LIGHTMAP_CHUNK_SIZE][LIGHTMAP_CHUNK_SIZE]) {
    float x = light.pos.x / (32.0f / LIGHTING_CELL_SUBDIVISIONS);
    float y = light.pos.y / (32.0f / LIGHTING_CELL_SUBDIVISIONS);
    float z = light.pos.z / 2.0f;
    int lm_x = (int)x - LIGHTMAXSPREAD;
    int lm_y = (int)y - LIGHTMAXSPREAD;
    int lm_z = (int)z - LIGHTMAXSPREAD * 2;

    // apply
    for (int llm_z = 0; llm_z < LIGHTMAP_CHUNK_SIZE; llm_z++) {
        for (int llm_y = 0; llm_y < LIGHTMAP_CHUNK_SIZE; llm_y++) {
            for (int llm_x = 0; llm_x < LIGHTMAP_CHUNK_SIZE; llm_x++) {
                int chlm_x = llm_x + target->x * LIGHTMAP_CHUNK_SIZE;
                int chlm_y = llm_y + target->y * LIGHTMAP_CHUNK_SIZE;
                int chlm_z = llm_z + target->z * LIGHTMAP_CHUNK_SIZE;
                int lp_offset_x = chlm_x - lm_x;
                int lp_offset_y = chlm_y - lm_y;
                int lp_offset_z = chlm_z - lm_z;

                if (lp_offset_x >= 0 && lp_offset_x < LIGHTMAXSPREAD * 2 && lp_offset_y >= 0 && lp_offset_y < LIGHTMAXSPREAD * 2 && lp_offset_z >= 0 && lp_offset_z < LIGHTMAXSPREAD * 4) {
                    lighting_add(&target_data[llm_z][llm_y][llm_x], map[lp_offset_z][lp_offset_y][lp_offset_x]);
                }
            }
        }
    }
}
/*
// expand + apply a light
static void light_expand(lighting_light light, lighting_chunk* target, lighting_value target_data[LIGHTMAP_CHUNK_SIZE][LIGHTMAP_CHUNK_SIZE][LIGHTMAP_CHUNK_SIZE]) {
    lighting_value map[LIGHTMAXSPREAD * 4 + 1][LIGHTMAXSPREAD * 2 + 1][LIGHTMAXSPREAD * 2 + 1];
    light_expand_to_map(light, map);
    light_expansion_apply(light, map, target, target_data);
}
*/
/*
// cast a ray from a 3d world position @a (light source position) to lighting tile
// @return 
static lighting_value FASTCALL lighting_raycast(lighting_value color, const rct_xyz32 light_source_pos, const rct_xyz16 lightmap_texel) {
    float x = light_source_pos.x / (32.0f / LIGHTING_CELL_SUBDIVISIONS);
    float y = light_source_pos.y / (32.0f / LIGHTING_CELL_SUBDIVISIONS);
    float z = light_source_pos.z / 2.0f;
    float dx = (lightmap_texel.x + .5) - x;
    float dy = (lightmap_texel.y + .5) - y;
    float dz = (lightmap_texel.z + .5) - z;

    for (int px = Math::Min(lightmap_texel.x + 1, (int)ceil(x)); px <= Math::Max((sint16)floor(x), lightmap_texel.x); px++) {
        float multiplier = (px - x) / dx;
        float py = dy * multiplier + y - .5;
        float pz = dz * multiplier + z - .5;
        int affectorY = py;
        int affectorZ = pz;
        float progy = py - affectorY;
        float progz = pz - affectorZ;
        lighting_value v00 = lightingAffectorsX[LAIDX(affectorY, px, affectorZ)];
        lighting_value v01 = lightingAffectorsX[LAIDX(affectorY + 1, px, affectorZ)];
        lighting_value v10 = lightingAffectorsX[LAIDX(affectorY, px, affectorZ + 1)];
        lighting_value v11 = lightingAffectorsX[LAIDX(affectorY + 1, px, affectorZ + 1)];
        lighting_value v0 = interpolate_lighting(v00, v01, progy);
        lighting_value v1 = interpolate_lighting(v10, v11, progy);
        lighting_multiply(&color, interpolate_lighting(v0, v1, progz));
    }

    for (int py = Math::Min(lightmap_texel.y + 1, (int)ceil(y)); py <= Math::Max((sint16)floor(y), lightmap_texel.y); py++) {
        float multiplier = (py - y) / dy;
        float px = dx * multiplier + x - .5;
        float pz = dz * multiplier + z - .5;
        int affectorX = px;
        int affectorZ = pz;
        float progx = px - affectorX;
        float progz = pz - affectorZ;
        lighting_value v00 = lightingAffectorsY[LAIDX(py, affectorX, affectorZ)];
        lighting_value v01 = lightingAffectorsY[LAIDX(py, affectorX + 1, affectorZ)];
        lighting_value v10 = lightingAffectorsY[LAIDX(py, affectorX, affectorZ + 1)];
        lighting_value v11 = lightingAffectorsY[LAIDX(py, affectorX + 1, affectorZ + 1)];
        lighting_value v0 = interpolate_lighting(v00, v01, progx);
        lighting_value v1 = interpolate_lighting(v10, v11, progx);
        lighting_multiply(&color, interpolate_lighting(v0, v1, progz));
    }

    for (int pz = Math::Min(lightmap_texel.z + 1, (int)ceil(z)); pz <= Math::Max((sint16)floor(z), lightmap_texel.z); pz++) {
        float multiplier = (pz - z) / dz;
        float px = dx * multiplier + x - .5;
        float py = dy * multiplier + y - .5;
        int affectorX = px;
        int affectorY = py;
        float progx = px - affectorX;
        float progy = py - affectorY;
        lighting_value v00 = lightingAffectorsZ[LAIDX(affectorY, affectorX, pz)];
        lighting_value v01 = lightingAffectorsZ[LAIDX(affectorY, affectorX + 1, pz)];
        lighting_value v10 = lightingAffectorsZ[LAIDX(affectorY + 1, affectorX, pz)];
        lighting_value v11 = lightingAffectorsZ[LAIDX(affectorY + 1, affectorX + 1, pz)];
        lighting_value v0 = interpolate_lighting(v00, v01, progx);
        lighting_value v1 = interpolate_lighting(v10, v11, progx);
        lighting_multiply(&color, interpolate_lighting(v0, v1, progy));
    }

    return color;
}
*/

// inserts a static light into the chunks this light can reach
static void lighting_insert_static_light(const lighting_light light) {
    int range = 11;
    sint32 lm_x = light.map_x * LIGHTING_CELL_SUBDIVISIONS;
    sint32 lm_y = light.map_y * LIGHTING_CELL_SUBDIVISIONS;
    sint32 lm_z = light.pos.z / 2;
    CHUNKRANGEITRXYZ(lm_x, lm_y, lm_z, sx, sy, sz, range) {
        lighting_chunk* chunk = &LIGHTINGCHUNK(sz, sy, sx);
        // TODO: bounds check
        lighting_chunk_static_light& light_slot = chunk->static_lights[chunk->static_lights_count++];
        light_slot.light = light;
        light_slot.is_drawn = false;
    }
}

void lighting_invalidate_at(sint32 wx, sint32 wy) {
    // remove static lights at this position
    // iterate chunks lights could reach, find lights in this column, remove them
    int range = 11;
    sint32 lm_x = wx * LIGHTING_CELL_SUBDIVISIONS;
    sint32 lm_y = wy * LIGHTING_CELL_SUBDIVISIONS;
    for (int sz = 0; sz < LIGHTMAP_CHUNKS_Z; sz++) {
        CHUNKRANGEITRXY(lm_x, lm_y, sx, sy, range) {
            lighting_chunk* chunk = &LIGHTINGCHUNK(sz, sy, sx);
            for (size_t lidx = 0; lidx < chunk->static_lights_count; lidx++) {
                if (chunk->static_lights[lidx].light.map_x == wx && chunk->static_lights[lidx].light.map_y == wy) {
                    chunk->static_lights[lidx] = chunk->static_lights[chunk->static_lights_count - 1];
                    chunk->static_lights_count--;
                    lidx--;
                }
            }

            memset(chunk->data_static, 0, sizeof(chunk->data_static));
            memcpy(chunk->data_skylight_static, chunk->data_skylight, sizeof(chunk->data_skylight));
            for (size_t light_idx = 0; light_idx < chunk->static_lights_count; light_idx++) {
                chunk->static_lights[light_idx].is_drawn = false;
            }

            outdated_static.push(chunk);
            {
                std::lock_guard<std::mutex> lock(outdated_gpu_mutex);
                outdated_gpu.push(chunk);
            }
        }
    }

    // iterate column
    rct_map_element* map_element = map_get_first_element_at(wx, wy);
    if (map_element) {
        do {
            switch (map_element_get_type(map_element)) {
            case MAP_ELEMENT_TYPE_PATH: {
                if (footpath_element_has_path_scenery(map_element) && !(map_element->flags & MAP_ELEMENT_FLAG_BROKEN)) {
                    // test
                    rct_scenery_entry *sceneryEntry = get_footpath_item_entry(footpath_element_get_path_scenery_index(map_element));
                    sint32 x = wx * 32 + 16;
                    sint32 y = wy * 32 + 16;
                    if (sceneryEntry->path_bit.flags & PATH_BIT_FLAG_LAMP) {
                        int z = map_element->base_height * 2 + 6;
                        if (!(map_element->properties.path.edges & (1 << 0))) {
                            lighting_insert_static_light({ { x - 14, y, z }, lit, wx, wy });
                        }
                        if (!(map_element->properties.path.edges & (1 << 1))) {
                            lighting_insert_static_light({ { x, y + 14, z }, lit, wx, wy });
                        }
                        if (!(map_element->properties.path.edges & (1 << 2))) {
                            lighting_insert_static_light({ { x + 14, y, z }, lit, wx, wy });
                        }
                        if (!(map_element->properties.path.edges & (1 << 3))) {
                            lighting_insert_static_light({ { x, y - 14, z }, lit, wx, wy });
                        }
                    }
                }
                break;
            }
            }
        } while (!map_element_is_last_for_tile(map_element++));
    }

    // revert values to lit...
    for (int lm_z = 0; lm_z < LIGHTMAP_SIZE_Z; lm_z++) {
        for (int ulm_x = lm_x; ulm_x <= lm_x + LIGHTING_CELL_SUBDIVISIONS; ulm_x++) {
            SUBCELLITR(sy, lm_y) lightingAffectorsX[LAIDX(sy, ulm_x, lm_z)] = lit;
        }
        for (int ulm_y = lm_y; ulm_y <= lm_y + LIGHTING_CELL_SUBDIVISIONS; ulm_y++) {
            SUBCELLITR(sx, lm_x) lightingAffectorsY[LAIDX(ulm_y, sx, lm_z)] = lit;
        }
        SUBCELLITR(sy, lm_y) SUBCELLITR(sx, lm_x) lightingAffectorsZ[LAIDX(sy, sx, lm_z)] = lit;
    }

    // queue rebuilding affectors
    SUBCELLITR(sy, lm_y) affectorRecomputeQueue[sy][lm_x] = 0b1111;
    SUBCELLITR(sx, lm_x) affectorRecomputeQueue[lm_y][sx] = 0b1111;

    if (lm_x > 0) { // east
        SUBCELLITR(sy, lm_y) affectorRecomputeQueue[sy][lm_x - 1] |= 0b0100;
    }
    if (lm_y > 0) { // north
        SUBCELLITR(sx, lm_x) affectorRecomputeQueue[lm_y - 1][sx] |= 0b0010;
    }
    if (lm_x < LIGHTMAP_SIZE_X - 2) { // east
        SUBCELLITR(sy, lm_y) affectorRecomputeQueue[sy][lm_x + LIGHTING_CELL_SUBDIVISIONS] |= 0b0001;
    }
    if (lm_y < LIGHTMAP_SIZE_Y - 2) { // south
        SUBCELLITR(sx, lm_x) affectorRecomputeQueue[lm_y + LIGHTING_CELL_SUBDIVISIONS][sx] |= 0b1000;
    }
}

void lighting_invalidate_around(sint32 wx, sint32 wy) {
    lighting_invalidate_at(wx, wy);
    if (wx < LIGHTMAP_SIZE_X - 1) lighting_invalidate_at(wx + 1, wy);
    if (wy < LIGHTMAP_SIZE_Y - 1) lighting_invalidate_at(wx, wy + 1);
    if (wx > 0) lighting_invalidate_at(wx - 1, wy);
    if (wy > 0) lighting_invalidate_at(wx, wy - 1);
}

static void lighting_enqueue_next_skylight_batch();
void lighting_set_skylight_direction(float direction[3]);
static void lighting_worker_thread();
void lighting_init() {
    lighting_cleanup();

    lightingChunks = (lighting_chunk*)malloc(sizeof(lighting_chunk) * LIGHTMAP_CHUNKS_X * LIGHTMAP_CHUNKS_Y * LIGHTMAP_CHUNKS_Z);
    // TODO: this should honestly really be a power of two size for fast multiplication for indexing...
    lightingAffectorsX = (lighting_value*)malloc(sizeof(lighting_value) * (LIGHTMAP_SIZE_X + 1) * (LIGHTMAP_SIZE_Y + 1) * (LIGHTMAP_SIZE_Z + 1));
    lightingAffectorsY = (lighting_value*)malloc(sizeof(lighting_value) * (LIGHTMAP_SIZE_X + 1) * (LIGHTMAP_SIZE_Y + 1) * (LIGHTMAP_SIZE_Z + 1));
    lightingAffectorsZ = (lighting_value*)malloc(sizeof(lighting_value) * (LIGHTMAP_SIZE_X + 1) * (LIGHTMAP_SIZE_Y + 1) * (LIGHTMAP_SIZE_Z + 1));
	{
		std::lock_guard<std::mutex> lock(outdated_gpu_mutex);
		outdated_gpu.clear();
	}
    outdated_static.clear();
    {
        std::lock_guard<std::mutex> lock(outdated_skylight_mutex);
        outdated_skylight.clear();
    }
    std::unordered_set<lighting_chunk*>().swap(dynamic_chunks);

    // reset affectors to 1^3
    for (int z = 0; z < LIGHTMAP_SIZE_Z; z++) {
        for (int y = 0; y < LIGHTMAP_SIZE_Y; y++) {
            for (int x = 0; x < LIGHTMAP_SIZE_X; x++) {
                lightingAffectorsX[LAIDX(y, x, z)] = lit;
                lightingAffectorsY[LAIDX(y, x, z)] = lit;
                lightingAffectorsZ[LAIDX(y, x, z)] = lit;
            }
        }
    }

    // init chunks
    for (int z = 0; z < LIGHTMAP_CHUNKS_Z; z++) {
        for (int y = 0; y < LIGHTMAP_CHUNKS_Y; y++) {
            for (int x = 0; x < LIGHTMAP_CHUNKS_X; x++) {
                memset(LIGHTINGCHUNK(z, y, x).data_skylight, 0, sizeof(LIGHTINGCHUNK(z, y, x).data_skylight));
                LIGHTINGCHUNK(z, y, x).static_lights_count = 0;
                LIGHTINGCHUNK(z, y, x).x = x;
                LIGHTINGCHUNK(z, y, x).y = y;
                LIGHTINGCHUNK(z, y, x).z = z;
            }
        }
    }

    skylight_delta = { 0, 0, 0 }; // must reset, pointers are no longer valid
    float new_skylight_direction[3] = { 0.2f, -0.4f, -0.4f };
    lighting_set_skylight_direction(new_skylight_direction);

    skylight_batch_current = -1;
    lighting_enqueue_next_skylight_batch();

    if (worker_threads.size() == 0) {
        worker_threads_continue.store(true);
        worker_threads.push_back(std::thread(lighting_worker_thread));
    }

    lighting_reset();
}

void lighting_invalidate_all() {
    // invalidate/recompute all columns ((re)loads all lights on the map)
    for (int y = 0; y < MAXIMUM_MAP_SIZE_PRACTICAL - 1; y++) {
        for (int x = 0; x < MAXIMUM_MAP_SIZE_PRACTICAL - 1; x++) {
            lighting_invalidate_at(x, y);
        }
    }
}

void lighting_cleanup() {
    log_info("Cleaning");
    worker_threads_continue.store(false);
    for (std::thread& thread : worker_threads) {
        thread.join();
    }
    log_info("Done");
    worker_threads.clear();

    free(lightingChunks);
    free(lightingAffectorsX);
    free(lightingAffectorsY);
    free(lightingAffectorsZ);
}

void lighting_set_skylight_direction(float direction[3]) {
    memcpy(skylight_direction, direction, sizeof(float) * 3);
    rct_xyz16 delta = { static_cast<sint16>(direction[0] > 0 ? -1 : 1), static_cast<sint16>(direction[1] > 0 ? -1 : 1), static_cast<sint16>(direction[2] > 0 ? -1 : 1) };

    if (delta.x != skylight_delta.x || delta.y != skylight_delta.y || delta.z != skylight_delta.z) {
        // rebuild chunk iterators...
        for (int dist = 0; dist < LIGHTMAP_CHUNKS_X + LIGHTMAP_CHUNKS_Y + LIGHTMAP_CHUNKS_Z; dist++) {
            skylight_batch[dist].clear();
        }

        rct_xyz16 sourceChunk = { static_cast<sint16>(delta.x == 1 ? LIGHTMAP_CHUNKS_X - 1 : 0), static_cast<sint16>(delta.y == 1 ? LIGHTMAP_CHUNKS_Y - 1 : 0), static_cast<sint16>(delta.z == 1 ? LIGHTMAP_CHUNKS_Z - 1 : 0) };
        log_info("start at %d %d %d", sourceChunk.x, sourceChunk.y, sourceChunk.z);

        for (int z = 0; z < LIGHTMAP_CHUNKS_Z; z++) {
            for (int y = 0; y < LIGHTMAP_CHUNKS_Y; y++) {
                for (int x = 0; x < LIGHTMAP_CHUNKS_X; x++) {
                    int dist = abs(sourceChunk.x - x) + abs(sourceChunk.y - y) + abs(sourceChunk.z - z);
                    skylight_batch[dist].emplace_back(&LIGHTINGCHUNK(z, y, x));
                }
            }
        }

        // rebuild cell iterator...
        rct_xyz16 sourceCell = { static_cast<sint16>(delta.x == 1 ? LIGHTMAP_CHUNK_SIZE - 1 : 0), static_cast<sint16>(delta.y == 1 ? LIGHTMAP_CHUNK_SIZE - 1 : 0), static_cast<sint16>(delta.z == 1 ? LIGHTMAP_CHUNK_SIZE - 1 : 0) };

        int itr_queue_build_pos = 0;
        for (sint16 dist = 0; dist <= LIGHTMAP_CHUNK_SIZE + LIGHTMAP_CHUNK_SIZE + LIGHTMAP_CHUNK_SIZE; dist++) {
            for (sint16 x = 0; x < LIGHTMAP_CHUNK_SIZE; x++)
                for (sint16 y = 0; y < LIGHTMAP_CHUNK_SIZE; y++)
                    for (sint16 z = 0; z < LIGHTMAP_CHUNK_SIZE; z++) {
                        int thisdist = abs(sourceCell.x - x) + abs(sourceCell.y - y) + abs(sourceCell.z - z);
                        if (thisdist == dist) skylight_cell_itr[itr_queue_build_pos++] = { x, y, z };
                    }
        }

        skylight_delta = delta;
    }

    skylight_batch_current = 0;

    log_info("Set skylight %f %f %f", direction[0], direction[1], direction[2]);
}

#pragma region Colormap
// stolen from opengldrawingengine
static const float TransparentColourTable[144 - 44][3] =
{
    { 0.7f, 0.8f, 0.8f }, // 44
    { 0.7f, 0.8f, 0.8f },
    { 0.3f, 0.4f, 0.4f },
    { 0.2f, 0.3f, 0.3f },
    { 0.1f, 0.2f, 0.2f },
    { 0.4f, 0.5f, 0.5f },
    { 0.3f, 0.4f, 0.4f },
    { 0.4f, 0.5f, 0.5f },
    { 0.4f, 0.5f, 0.5f },
    { 0.3f, 0.4f, 0.4f },
    { 0.6f, 0.7f, 0.7f },
    { 0.3f, 0.5f, 0.9f },
    { 0.1f, 0.3f, 0.8f },
    { 0.5f, 0.7f, 0.9f },
    { 0.6f, 0.2f, 0.2f },
    { 0.5f, 0.1f, 0.1f },
    { 0.8f, 0.4f, 0.4f },
    { 0.3f, 0.5f, 0.4f },
    { 0.2f, 0.4f, 0.2f },
    { 0.5f, 0.7f, 0.5f },
    { 0.5f, 0.5f, 0.7f },
    { 0.3f, 0.3f, 0.5f },
    { 0.6f, 0.6f, 0.8f },
    { 0.5f, 0.5f, 0.2f },
    { 0.4f, 0.4f, 0.1f },
    { 0.7f, 0.7f, 0.4f },
    { 0.7f, 0.5f, 0.3f },
    { 0.6f, 0.4f, 0.2f },
    { 0.8f, 0.7f, 0.4f },
    { 0.8f, 0.7f, 0.1f },
    { 0.7f, 0.4f, 0.0f },
    { 1.0f, 0.9f, 0.2f },
    { 0.4f, 0.6f, 0.2f },
    { 0.3f, 0.4f, 0.2f },
    { 0.5f, 0.7f, 0.3f },
    { 0.5f, 0.6f, 0.4f },
    { 0.4f, 0.4f, 0.3f },
    { 0.7f, 0.8f, 0.5f },
    { 0.3f, 0.7f, 0.2f },
    { 0.2f, 0.6f, 0.0f },
    { 0.4f, 0.8f, 0.3f },
    { 0.8f, 0.5f, 0.4f },
    { 0.7f, 0.4f, 0.3f },
    { 0.9f, 0.7f, 0.5f },
    { 0.5f, 0.3f, 0.7f },
    { 0.4f, 0.2f, 0.6f },
    { 0.7f, 0.5f, 0.8f },
    { 0.9f, 0.0f, 0.0f },
    { 0.7f, 0.0f, 0.0f },
    { 1.0f, 0.3f, 0.3f },
    { 1.0f, 0.4f, 0.1f },
    { 0.9f, 0.3f, 0.0f },
    { 1.0f, 0.6f, 0.3f },
    { 0.2f, 0.6f, 0.6f },
    { 0.0f, 0.4f, 0.4f },
    { 0.4f, 0.7f, 0.7f },
    { 0.9f, 0.2f, 0.6f },
    { 0.6f, 0.1f, 0.4f },
    { 1.0f, 0.5f, 0.7f },
    { 0.6f, 0.5f, 0.4f },
    { 0.4f, 0.3f, 0.2f },
    { 0.7f, 0.7f, 0.6f },
    { 0.9f, 0.6f, 0.6f },
    { 0.8f, 0.5f, 0.5f },
    { 1.0f, 0.7f, 0.7f },
    { 0.7f, 0.8f, 0.8f },
    { 0.5f, 0.6f, 0.6f },
    { 0.9f, 1.0f, 1.0f },
    { 0.2f, 0.3f, 0.3f },
    { 0.4f, 0.5f, 0.5f },
    { 0.7f, 0.8f, 0.8f },
    { 0.2f, 0.3f, 0.5f },
    { 0.5f, 0.5f, 0.7f },
    { 0.5f, 0.3f, 0.7f },
    { 0.1f, 0.3f, 0.7f },
    { 0.3f, 0.5f, 0.9f },
    { 0.6f, 0.8f, 1.0f },
    { 0.2f, 0.6f, 0.6f },
    { 0.5f, 0.8f, 0.8f },
    { 0.1f, 0.5f, 0.0f },
    { 0.3f, 0.5f, 0.4f },
    { 0.4f, 0.6f, 0.2f },
    { 0.3f, 0.7f, 0.2f },
    { 0.5f, 0.6f, 0.4f },
    { 0.5f, 0.5f, 0.2f },
    { 1.0f, 0.9f, 0.2f },
    { 0.8f, 0.7f, 0.1f },
    { 0.6f, 0.3f, 0.0f },
    { 1.0f, 0.4f, 0.1f },
    { 0.7f, 0.3f, 0.0f },
    { 0.7f, 0.5f, 0.3f },
    { 0.5f, 0.3f, 0.1f },
    { 0.5f, 0.4f, 0.3f },
    { 0.8f, 0.5f, 0.4f },
    { 0.6f, 0.2f, 0.2f },
    { 0.6f, 0.0f, 0.0f },
    { 0.9f, 0.0f, 0.0f },
    { 0.6f, 0.1f, 0.3f },
    { 0.9f, 0.2f, 0.6f },
    { 0.9f, 0.6f, 0.6f },
};
#pragma endregion Colormap
/*
static void lighting_static_light_cast(lighting_value* target_value, lighting_light light, sint16 px, sint16 py, sint16 pz) {
    //sint32 range = 11;
    sint16 w_x = px * 16 + 8;
    sint16 w_y = py * 16 + 8;
    sint16 w_z = pz * 2 + 1;
    float distpot = sqrt((w_x - light.pos.x)*(w_x - light.pos.x) + (w_y - light.pos.y)*(w_y - light.pos.y) + (w_z - light.pos.z)*(w_z - light.pos.z) * 4 * 4);
    float intensity = 900.0f / (distpot*distpot);
    if (intensity > 0) {
        if (intensity > 0.5f) intensity = 0.5f;
        rct_xyz16 target = { px, py, pz };
        intensity *= 35;
        lighting_value source_value = { (uint8)intensity, (uint8)intensity, (uint8)intensity };
        lighting_multiply(&source_value, lightlit);
        lighting_add(target_value, lighting_raycast(source_value, light.pos, target));
    }
}
*/
static void lighting_update_affectors() {
    for (int y = 0; y < LIGHTMAP_SIZE_Y; y++) {
        for (int x = 0; x < LIGHTMAP_SIZE_X; x++) {
            uint8 dirs = affectorRecomputeQueue[y][x];
            if (dirs) {
                rct_map_element* map_element = map_get_first_element_at(x / LIGHTING_CELL_SUBDIVISIONS, y / LIGHTING_CELL_SUBDIVISIONS);

                uint8 quadrant_offset_x = (x / (LIGHTING_CELL_SUBDIVISIONS / 2)) % 2;
                uint8 quadrant_offset_y = (y / (LIGHTING_CELL_SUBDIVISIONS / 2)) % 2;
                uint8 quadrant_mask;
                if (!quadrant_offset_x && !quadrant_offset_y) quadrant_mask = 1 << 2;
                else if (quadrant_offset_x && !quadrant_offset_y) quadrant_mask = 1 << 1;
                else if (!quadrant_offset_x && quadrant_offset_y) quadrant_mask = 1 << 3;
                else quadrant_mask = 1 << 0;

                // test
                if (map_element) {
                    do {
                        switch (map_element_get_type(map_element))
                        {
                        case MAP_ELEMENT_TYPE_SURFACE: {
                            for (int z = 0; z < map_element->base_height - 1; z++) {
                                lightingAffectorsX[LAIDX(y, x, z)] = black;
                                lightingAffectorsY[LAIDX(y, x, z)] = black;
                                lightingAffectorsX[LAIDX(y, x + 1, z)] = black;
                                lightingAffectorsY[LAIDX(y + 1, x, z)] = black;
                                lightingAffectorsZ[LAIDX(y, x, z)] = black;
                                lightingAffectorsZ[LAIDX(y, x, z + 1)] = black;
                            }
                            break;
                        }
                        case MAP_ELEMENT_TYPE_SCENERY: {
                            if (!(map_element->flags & quadrant_mask)) continue;
                            for (int z = map_element->base_height - 1; z < map_element->clearance_height - 1; z++) {
                                if (map_element->clearance_height - map_element->base_height > 5)
                                {
                                    // probably a tree or so
                                    lighting_multiply(&lightingAffectorsX[LAIDX(y, x, z)], dimmedblackside);
                                    lighting_multiply(&lightingAffectorsY[LAIDX(y, x, z)], dimmedblackside);
                                    lighting_multiply(&lightingAffectorsX[LAIDX(y, x + 1, z)], dimmedblackside);
                                    lighting_multiply(&lightingAffectorsY[LAIDX(y + 1, x, z)], dimmedblackside);
                                    lighting_multiply(&lightingAffectorsZ[LAIDX(y, x, z)], dimmedblackvside);
                                    lighting_multiply(&lightingAffectorsZ[LAIDX(y, x, z + 1)], dimmedblackvside);
                                }
                                else
                                {
                                    lightingAffectorsX[LAIDX(y, x, z)] = black;
                                    lightingAffectorsY[LAIDX(y, x, z)] = black;
                                    lightingAffectorsX[LAIDX(y, x + 1, z)] = black;
                                    lightingAffectorsY[LAIDX(y + 1, x, z)] = black;
                                    lightingAffectorsZ[LAIDX(y, x, z)] = black;
                                    lightingAffectorsZ[LAIDX(y, x, z + 1)] = black;
                                }
                            }
                            break;
                        }
                        case MAP_ELEMENT_TYPE_TRACK: {
                            if (!(map_element->flags & quadrant_mask)) continue;
                            for (int z = map_element->base_height - 1; z < map_element->clearance_height - 3; z++) {
                                // TODO check side flag if it should be updated
                                lighting_multiply(&lightingAffectorsX[LAIDX(y, x, z)], dimmedblackside);
                                lighting_multiply(&lightingAffectorsY[LAIDX(y, x, z)], dimmedblackside);
                                lighting_multiply(&lightingAffectorsX[LAIDX(y, x + 1, z)], dimmedblackside);
                                lighting_multiply(&lightingAffectorsY[LAIDX(y + 1, x, z)], dimmedblackside);
                                lighting_multiply(&lightingAffectorsZ[LAIDX(y, x, z)], dimmedblack);
                                lighting_multiply(&lightingAffectorsZ[LAIDX(y, x, z + 1)], dimmedblack);
                            }
                            break;
                        }
                        case MAP_ELEMENT_TYPE_PATH: {
                            int z = map_element->base_height - 1;
                            lightingAffectorsZ[LAIDX(y, x, z)] = black;
                            break;
                        }
                        case MAP_ELEMENT_TYPE_WALL: {
                            // do not apply if the wall its direction is not queued
                            if (!(dirs & (1 << map_element_get_direction(map_element)))) {
                                break;
                            }
                            for (int z = map_element->base_height - 1; z < map_element->clearance_height; z++) {
                                lighting_value affector = black;
                                if (map_element->properties.wall.type == 54) {
                                    uint8 color = map_element->properties.wall.colour_1;
                                    if (color >= 44 && color < 144) {
                                        affector.r = TransparentColourTable[color - 44][0] * 255;
                                        affector.g = TransparentColourTable[color - 44][1] * 255;
                                        affector.b = TransparentColourTable[color - 44][2] * 255;
                                    }
                                }
                                switch (map_element_get_direction(map_element)) {
                                case MAP_ELEMENT_DIRECTION_NORTH:
                                    if (y % LIGHTING_CELL_SUBDIVISIONS == LIGHTING_CELL_SUBDIVISIONS - 1) lighting_multiply(&lightingAffectorsY[LAIDX(y + LIGHTING_CELL_SUBDIVISIONS - 1, x, z)], affector);
                                    break;
                                case MAP_ELEMENT_DIRECTION_SOUTH:
                                    if (y % LIGHTING_CELL_SUBDIVISIONS == 0) lighting_multiply(&lightingAffectorsY[LAIDX(y, x, z)], affector);
                                    break;
                                case MAP_ELEMENT_DIRECTION_EAST:
                                    if (x % LIGHTING_CELL_SUBDIVISIONS == LIGHTING_CELL_SUBDIVISIONS - 1) lighting_multiply(&lightingAffectorsX[LAIDX(y, x + LIGHTING_CELL_SUBDIVISIONS - 1, z)], affector);
                                    break;
                                case MAP_ELEMENT_DIRECTION_WEST:
                                    if (x % LIGHTING_CELL_SUBDIVISIONS == 0) lighting_multiply(&lightingAffectorsX[LAIDX(y, x, z)], affector);
                                    break;
                                }
                            }
                            break;
                        }
                        }
                    } while (!map_element_is_last_for_tile(map_element++));
                }
                affectorRecomputeQueue[y][x] = 0;
            }
        }
    }
}

static void lighting_update_static_light(lighting_light& light) {
    int range = 11;
    sint32 lm_x = light.map_x * LIGHTING_CELL_SUBDIVISIONS;
    sint32 lm_y = light.map_y * LIGHTING_CELL_SUBDIVISIONS;
    sint32 lm_z = light.pos.z / 2;

    lighting_value map[LIGHTMAXSPREAD * 4 + 1][LIGHTMAXSPREAD * 2 + 1][LIGHTMAXSPREAD * 2 + 1];
    light_expand_to_map(light, map);

    CHUNKRANGEITRXYZ(lm_x, lm_y, lm_z, sx, sy, sz, range) {
        lighting_chunk* chunk = &LIGHTINGCHUNK(sz, sy, sx);
        
        // where's this light?
        for (size_t light_idx = 0; light_idx < chunk->static_lights_count; light_idx++) {
            // this isn't garantueed to work properly (two lights with equal data can be at the same position)
            if (!memcmp(&chunk->static_lights[light_idx].light, &light, sizeof(lighting_light))) {
                if (!chunk->static_lights[light_idx].is_drawn) {
                    light_expansion_apply(light, map, chunk, chunk->data_static);
                    light_expansion_apply(light, map, chunk, chunk->data_skylight_static); // TODO: applying to this can be done much more efficiently for sure
                    chunk->static_lights[light_idx].is_drawn = true;
                }
                break;
            }
        }
    }
}
/*
static void lighting_update_chunk(lighting_chunk* chunk) {
    for (int oz = LIGHTMAP_CHUNK_SIZE - 1; oz >= 0; oz--) {
        for (int oy = 0; oy < LIGHTMAP_CHUNK_SIZE; oy++) {
            for (int ox = 0; ox < LIGHTMAP_CHUNK_SIZE; ox++) {
                chunk->data_static[oz][oy][ox] = ambient;

                // initialize to skylight value
                chunk->data_static[oz][oy][ox] = chunk->skylight_carry[oy][ox];

                // update carry skylight
                lighting_value affector = lightingAffectorsZ[LAIDX(chunk->y*LIGHTMAP_CHUNK_SIZE + oy, chunk->x*LIGHTMAP_CHUNK_SIZE + ox, chunk->z*LIGHTMAP_CHUNK_SIZE + oz)];
                lighting_multiply(&chunk->skylight_carry[oy][ox], affector);
            }
        }
    }

    for (size_t lidx = 0; lidx < chunk->static_lights_count; lidx++) {
        // TODO: expansion data can be reused, which severely boosts performance
        light_expand(chunk->static_lights[lidx].light, chunk, chunk->data_static);
    }

    chunk->invalid = false;
}
*/
static void lighting_update_static(lighting_update_batch* updated_batch) {
    // TODO: this is not monotonic on Windows
    //clock_t max_end = clock() + LIGHTING_MAX_CLOCKS_PER_FRAME;

    // recompute invalid chunks until reaching a limit
    // start from the top to pass through skylights in the correct order
    /*for (int z = LIGHTMAP_CHUNKS_Z - 1; z >= 0; z--) {
        for (int y = 0; y < LIGHTMAP_CHUNKS_Y; y++) {
            for (int x = 0; x < LIGHTMAP_CHUNKS_X; x++) {
                // if it used to have dynamic lights, update always (to clean the dynamic lights)
                bool shouldUpdate = LIGHTINGCHUNK(z, y, x).has_dynamic_lights;
                LIGHTINGCHUNK(z, y, x).has_dynamic_lights = false;

                // invalidated static data?
                if (LIGHTINGCHUNK(z, y, x).invalid) {
                    // recompute this invalid chunk
                    lighting_chunk* chunk = &LIGHTINGCHUNK(z, y, x);
                    lighting_update_chunk(chunk);
                    shouldUpdate = true;
                }

                if (shouldUpdate) {
                    updated_batch->updated_chunks[updated_batch->update_count++] = &LIGHTINGCHUNK(z, y, x);

                    // exceeding max update count?
                    if (updated_batch->update_count >= LIGHTING_MAX_CHUNK_UPDATES_PER_FRAME) {
                        return;
                    }

                    // exceeding max time?
                    if (clock() > max_end) {
                        return;
                    }
                }
            }
        }
    }*/

    for (int i = 0; i < 100; i++) {
        if (outdated_static.empty()) break;
        lighting_chunk* chunk = outdated_static.frontpop();

        for (size_t lidx = 0; lidx < chunk->static_lights_count; lidx++) {
            if (!chunk->static_lights[lidx].is_drawn) {
                lighting_update_static_light(chunk->static_lights[lidx].light);
                assert(chunk->static_lights[lidx].is_drawn);
            }

            {
                std::lock_guard<std::mutex> lock(outdated_gpu_mutex);
                outdated_gpu.push(chunk);
            }
        }
    }
}
/*
static lighting_value* lighting_get_dynamic_texel(lighting_update_batch* updated_batch, int x, int y, int z) {
    int lm_z = z / LIGHTMAP_CHUNK_SIZE;
    int lm_y = y / LIGHTMAP_CHUNK_SIZE;
    int lm_x = x / LIGHTMAP_CHUNK_SIZE;

    if (lm_z < 0 || lm_y < 0 || lm_x < 0 || lm_z >= LIGHTMAP_CHUNKS_Z || lm_y >= LIGHTMAP_CHUNKS_Y || lm_x >= LIGHTMAP_CHUNKS_X) return NULL;

    lighting_chunk* chunk = &LIGHTINGCHUNK(lm_z, lm_y, lm_x);
    if (!chunk->has_dynamic_lights) {
        memcpy(chunk->data_dynamic, chunk->data, sizeof(chunk->data));
        chunk->has_dynamic_lights = true;

        updated_batch->updated_chunks[updated_batch->update_count++] = chunk;
    }

    return &chunk->data_dynamic[z % LIGHTMAP_CHUNK_SIZE][y % LIGHTMAP_CHUNK_SIZE][x % LIGHTMAP_CHUNK_SIZE];
}
*/
static void lighting_add_dynamic(lighting_update_batch* updated_batch, sint16 x, sint16 y, sint16 z) {
    int lm_x = (x * LIGHTING_CELL_SUBDIVISIONS) / 32;
    int lm_y = (y * LIGHTING_CELL_SUBDIVISIONS) / 32;
    int lm_z = z / 8;
    int range = 8;

    lighting_light light;
    light.pos = { x, y, z / 4 }; // TODO: not sure why this coordinate space is / 8 instead of / 2, which requires this random correction here

    lighting_value map[LIGHTMAXSPREAD * 4 + 1][LIGHTMAXSPREAD * 2 + 1][LIGHTMAXSPREAD * 2 + 1];
    light_expand_to_map(light, map);

    CHUNKRANGEITRXYZ(lm_x, lm_y, lm_z, ch_x, ch_y, ch_z, range) {
        lighting_chunk* chunk = &LIGHTINGCHUNK(ch_z, ch_y, ch_x);
        if (!chunk->has_dynamic_lights) {
            memcpy(chunk->data_dynamic, chunk->data_skylight_static, sizeof(chunk->data_static));
            chunk->has_dynamic_lights = true;

            dynamic_chunks.insert(chunk);
            //updated_batch->updated_chunks[updated_batch->update_count++] = chunk;
        }
        light_expansion_apply(light, map, chunk, chunk->data_dynamic);

        {
            std::lock_guard<std::mutex> lock(outdated_gpu_mutex);
            outdated_gpu.push(chunk);
        }
    }
}

static void lighting_update_dynamic(lighting_update_batch* updated_batch) {
    // TODO: this is not monotonic on Windows
    //clock_t max_end = clock() + LIGHTING_MAX_CLOCKS_PER_FRAME;

    //log_info("reg");

    uint16 spriteIndex = gSpriteListHead[SPRITE_LIST_TRAIN];
    while (spriteIndex != SPRITE_INDEX_NULL) {
        rct_vehicle * vehicle = &(get_sprite(spriteIndex)->vehicle);
        uint16 vehicleID = spriteIndex;
        spriteIndex = vehicle->next;

        if (vehicle->ride_subtype == RIDE_ENTRY_INDEX_NULL) {
            continue;
        }

        for (uint16 q = vehicleID; q != SPRITE_INDEX_NULL; ) {
            vehicle = GET_VEHICLE(q);

            vehicleID = q;
            if (vehicle->next_vehicle_on_train == q)
                break;
            q = vehicle->next_vehicle_on_train;

            sint16 place_x, place_y, place_z;

            place_x = vehicle->x;
            place_y = vehicle->y;
            place_z = vehicle->z;

            rct_ride *ride = get_ride(vehicle->ride);
            switch (ride->type) {
            case RIDE_TYPE_MONORAIL:
            case RIDE_TYPE_LOOPING_ROLLER_COASTER:
            case RIDE_TYPE_VIRGINIA_REEL:
            case RIDE_TYPE_MINE_RIDE:
            case RIDE_TYPE_MINE_TRAIN_COASTER:
            case RIDE_TYPE_WOODEN_ROLLER_COASTER:
            case RIDE_TYPE_MINIATURE_RAILWAY:
                if (vehicle == vehicle_get_head(vehicle)) {
                    lighting_add_dynamic(updated_batch, place_x, place_y, place_z);
                }
                break;
            case RIDE_TYPE_BOAT_RIDE:
            case RIDE_TYPE_WATER_COASTER:
                if (vehicle == vehicle_get_head(vehicle)) {
                    lighting_add_dynamic(updated_batch, place_x, place_y, place_z);
                }
                break;
            default:
                break;
            };
        }
    }

}

static lighting_value lighting_get_skylight_at(int w_x, int w_y, int w_z) {
    // out of bounds?
    if (w_x < 0 || w_y < 0 || w_z < 0 || w_x >= LIGHTMAP_SIZE_X || w_y >= LIGHTMAP_SIZE_Y || w_z >= LIGHTMAP_SIZE_Z)
        return ambient_sky;

    lighting_chunk& chunk = LIGHTINGCHUNK(w_z / LIGHTMAP_CHUNK_SIZE, w_y / LIGHTMAP_CHUNK_SIZE, w_x / LIGHTMAP_CHUNK_SIZE);
    return chunk.data_skylight[w_z % LIGHTMAP_CHUNK_SIZE][w_y % LIGHTMAP_CHUNK_SIZE][w_x % LIGHTMAP_CHUNK_SIZE];
}

static void lighting_enqueue_next_skylight_batch() {
    std::lock_guard<std::mutex> lock(outdated_skylight_mutex);

    do {
        skylight_batch_current = (skylight_batch_current + 1) % (LIGHTMAP_CHUNKS_X + LIGHTMAP_CHUNKS_Y + LIGHTMAP_CHUNKS_Z);

        for (lighting_chunk* chunk : skylight_batch[skylight_batch_current]) {
            outdated_skylight.push(chunk);
        }
    } while (skylight_batch[skylight_batch_current].size() == 0);

    skylight_batch_remaining = (uint8)skylight_batch[skylight_batch_current].size();
}

static void lighting_update_skylight(lighting_chunk* chunk) {
    for (int upd_idx = 0; upd_idx < LIGHTMAP_CHUNK_SIZE * LIGHTMAP_CHUNK_SIZE * LIGHTMAP_CHUNK_SIZE; upd_idx++) {
        const rct_xyz16& cell_coord = skylight_cell_itr[upd_idx];
        int w_x = chunk->x * LIGHTMAP_CHUNK_SIZE + cell_coord.x;
        int w_y = chunk->y * LIGHTMAP_CHUNK_SIZE + cell_coord.y;
        int w_z = chunk->z * LIGHTMAP_CHUNK_SIZE + cell_coord.z;

        lighting_value from_x = lighting_get_skylight_at(w_x + skylight_delta.x, w_y, w_z);
        lighting_value from_y = lighting_get_skylight_at(w_x, w_y + skylight_delta.y, w_z);
        lighting_value from_z = lighting_get_skylight_at(w_x, w_y, w_z + skylight_delta.z);

        lighting_multiply(&from_x, lightingAffectorsX[LAIDX(w_y, w_x + (skylight_delta.x < 0), w_z)]);
        lighting_multiply(&from_y, lightingAffectorsY[LAIDX(w_y + (skylight_delta.y < 0), w_x, w_z)]);
        lighting_multiply(&from_z, lightingAffectorsZ[LAIDX(w_y, w_x, w_z + (skylight_delta.z < 0))]);

        float fragx = fabs(skylight_direction[0]);
        float fragy = fabs(skylight_direction[1]);
        float fragz = fabs(skylight_direction[2]);

        // interpolate values
        lighting_value new_skylight_value = {
            (uint8)(from_x.r * fragx + from_y.r * fragy + from_z.r * fragz),
            (uint8)(from_x.g * fragx + from_y.g * fragy + from_z.g * fragz),
            (uint8)(from_x.b * fragx + from_y.b * fragy + from_z.b * fragz)
        };

        // TODO: refactor to something more efficient
        lighting_value static_value = chunk->data_static[cell_coord.z][cell_coord.y][cell_coord.x];

        chunk->data_skylight[cell_coord.z][cell_coord.y][cell_coord.x] = new_skylight_value;
        chunk->data_skylight_static[cell_coord.z][cell_coord.y][cell_coord.x] = {
            (uint8)(static_value.r + new_skylight_value.r),
            (uint8)(static_value.g + new_skylight_value.g),
            (uint8)(static_value.b + new_skylight_value.b),
        };
    }

    {
        std::lock_guard<std::mutex> lock(outdated_gpu_mutex);
        outdated_gpu.push(chunk);
    }
}

static void lighting_update_any_skylight() {
    lighting_chunk* chunk;
    {
        std::lock_guard<std::mutex> lock(outdated_skylight_mutex);

        if (outdated_skylight.empty()) return;

        chunk = outdated_skylight.frontpop();
    }

    lighting_update_skylight(chunk);
    
    if (--skylight_batch_remaining == 0) {
        lighting_enqueue_next_skylight_batch();
    }
}

static void lighting_worker_thread() {
    using namespace std::chrono_literals;

    while (worker_threads_continue.load()) { // TODO: may constantly read from cache?
        //std::unique_lock<std::mutex> lock(is_collecting_data_mutex);
        //is_collecting_data_cv.wait(lock, [] { return !is_collecting_data; } );
        lighting_update_any_skylight();
        std::this_thread::sleep_for(1ms);
        //lock.release();
    }
}

static lighting_update_batch lighting_update_internal() {

    // update all pending affectors first
    lighting_update_affectors();

    lighting_update_batch updated_batch;
    updated_batch.update_count = 0;

    /*for (int i = 0; i < 100; i++) {
        lighting_update_any_skylight();
    }*/

    // reset current dynamic chunks to static
    // must update at gpu too!
    {
        std::lock_guard<std::mutex> lock(outdated_gpu_mutex);
        for (lighting_chunk *chunk : dynamic_chunks) {
            chunk->has_dynamic_lights = false;
            outdated_gpu.push(chunk);
        }
    }
    std::unordered_set<lighting_chunk*>().swap(dynamic_chunks);

    lighting_update_static(&updated_batch);
    lighting_update_dynamic(&updated_batch);

    {
        std::lock_guard<std::mutex> lock(outdated_gpu_mutex);
        for (int i = 0; i < LIGHTING_MAX_CHUNK_UPDATES_PER_FRAME; i++) {
            if (outdated_gpu.empty()) break;

            lighting_chunk *chunk = outdated_gpu.frontpop();
            updated_batch.updated_chunks[updated_batch.update_count++] = chunk;
        }
    }

    updated_batch.updated_chunks[updated_batch.update_count] = NULL;

    return updated_batch;
}

void lighting_reset() {
    for (int y = 0; y < MAXIMUM_MAP_SIZE_PRACTICAL; y++) {
        for (int x = 0; x < MAXIMUM_MAP_SIZE_PRACTICAL; x++) {
            affectorRecomputeQueue[y][x] = 0b1111;
        }
    }
}

lighting_update_batch lighting_update() {
    return lighting_update_internal();
}
