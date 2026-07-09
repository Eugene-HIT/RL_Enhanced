cmake_minimum_required(VERSION 3.8)
project(rl_enhanced_gz_scene_timelapse)

find_package(ament_cmake REQUIRED)
find_package(gz-sim8 REQUIRED)
find_package(gz-plugin2 REQUIRED)
find_package(gz-math7 REQUIRED)

add_library(smooth_playback_system SHARED
  src/smooth_playback_system.cc
)
target_compile_features(smooth_playback_system PUBLIC cxx_std_17)
target_link_libraries(smooth_playback_system
  PRIVATE
    gz-sim8::gz-sim8
    gz-plugin2::gz-plugin2
    gz-math7::gz-math7
)

install(
  TARGETS smooth_playback_system
  LIBRARY DESTINATION lib
)

install(
  DIRECTORY launch worlds meshes rviz models scripts
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/generate_fixed_scene_world.py
    scripts/generate_rescue_world.py
    scripts/prepare_inference_playback_csv.py
    scripts/playback_payload_trajectory.py
    scripts/prepare_playback_csv.py
    scripts/spawn_timelapse_snapshots.py
    scripts/rviz_payload_playback.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
