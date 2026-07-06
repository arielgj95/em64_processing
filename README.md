# EM64 Processing

This repository contains a small set of Python scripts for working with Eigenmike 64 recordings, ambisonic audio, and 360 video. The main goal is to turn raw EM64 captures into aligned audio/video material and acoustic visualizations that make sound direction visible.

The scripts are research and production helpers rather than a polished package. They expect real recording files to be supplied when you run them, and the large local media outputs are intentionally kept out of git.

## What Is In The Repo

`audio-video_processing.py` aligns EM64 audio with a camera video by detecting a clap, trimming the files to the same start point, and producing acoustic overlays on top of the camera footage.

`acoustic_map.py` builds a 2D acoustic map video from HOA and raw Eigenmike recordings. It supports DAS, MAXRE, and MVDR map modes, then renders the result with binaural audio.

`3D-audio_processing.py` projects acoustic energy onto a room point cloud. It can show an interactive frame-by-frame viewer and export a 360 equirectangular point-cloud video.

`create_360_video.py` stacks an acoustic 360 render with a Ricoh 360 video after clap alignment. It is useful for comparing the acoustic visualization with the original camera view.

`em64_geom.csv` stores the microphone geometry and quadrature weights used by the beamforming code.

`AKO536081622_1_processed.sofa` is the HRTF file used for binaural rendering.

## How To Use It

Run the scripts from the repository root and pass your recording files as command-line arguments. The scripts no longer contain personal machine paths, so the same checkout can be used on another workstation as long as the required recordings are available locally.

Common inputs are:

- a raw 64-channel Eigenmike WAV file
- a HOA WAV file
- a camera or 360 video file
- the EM64 geometry CSV
- the SOFA HRTF file when binaural rendering is needed

Generated videos, aligned WAV files, point clouds, temporary render files, IDE settings, and scratch scripts are ignored by git. Keep those files local unless there is a specific reason to version them.

## Notes

The scripts depend on the scientific Python audio/video stack used in the code, including NumPy, SciPy, SoundFile, MoviePy, OpenCV, Open3D, Plotly, and related helpers. Some operations are heavy and can take a long time on full-resolution recordings.
