#include "../Scenes/Math/ConwayScene.h"
#include "../IO/Latex.h"

extern "C" void cuda_copy_pixels_to_host(uint32_t* h_pixels, int size, uint32_t* d_pixels);

void render_video() {
    ScalingParams sp(ivec2(1000, 1000));
    shared_ptr<DevicePointer> ourdemo = latex_to_gpu_pix("\\text{I really} \\\\\\\\ \\text{loved your} \\\\\\\\ \\text{crochet talk!}", sp);
    Pixels env(ivec2(1000, 1000));
    // ScalingParams fits the SVG inside the box rather than filling it, so the
    // rendered bitmap is usually far shorter than 1000 rows. Copy what is actually
    // there and leave the rest of the environment blank; asking for the whole box
    // reads past the end of the allocation.
    const ivec2 text_wh = ourdemo->get_wh();
    const int env_size = env.wh.x * env.wh.y;
    const int text_size = text_wh.x * text_wh.y;
    cuda_copy_pixels_to_host(env.pixels.data(), text_size < env_size ? text_size : env_size,
                             ourdemo->get_ptr());
    ConwayScene cs(ivec2(20000,20000), env);

    cs.manager.set("zoom", "-2");
    cs.manager.set("center_x", "-1000");// {t} .2 * cos 200 * +");
    cs.manager.set("center_y", "-24000");// {t} .2 * sin 200 * +");
    cs.manager.set("ticks_opacity", "0");

    stage_macroblock(SilenceBlock(10), 5);
    cs.render_microblock();
    cs.manager.transition(MICRO, "zoom", "-11");
    cs.render_microblock();
    cs.render_microblock();
    cs.manager.transition(MICRO, "zoom", "-2");
    cs.render_microblock();
    cs.render_microblock();
}
