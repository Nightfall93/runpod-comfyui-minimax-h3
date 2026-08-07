MINIMAX H3 PROMPT FORMULAS
==========================

These turn ChatGPT or Claude into a writer that produces prompts for the MiniMax H3
video model in ComfyUI. You describe the video you want, it hands you a finished prompt
you can paste straight into the H3 node.


PICK YOUR FOLDER
----------------

ChatGPT\   Two files per model. Open "HOW TO USE.txt" inside.
Claude\    One file per model.  Open "HOW TO USE.txt" inside.


PICK YOUR MODEL
---------------

FL2VA   minimax_h3_fl2va_*.safetensors
        Text to video, first frame, first and last frame, last frame.

REF2VA  minimax_h3_ref2va_*.safetensors
        Reference mode. You supply reference images, videos or audio that define a
        character, a place, a style, a camera move, a voice or a soundtrack.

Match this to the model you actually loaded in ComfyUI.


THEN WHAT
---------

Describe your video, or attach your image, and ask for a prompt. You get back one code
block. Copy it into the H3 prompt field in ComfyUI.

For first-and-last-frame and last-frame videos it will ask you how many seconds the
video is, because the reference image has to land on an exact time. Answer with a
number and it will write the prompt.

Tell it the duration if you know it. The prompt is paced to the clip length, so a
prompt written for 5 seconds is not the same as one written for 10.


BUILT FROM THE OFFICIAL SPEC
----------------------------

Every rule here was checked against MiniMax's own prompt writing guides:
https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/docs
