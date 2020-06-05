:

export CUDA_VISIBLE_DEVICES="-1"

#stdbuf -oL nohup sh -c  'nice -19 sh run_all.sh ALL --background --nogpu' > out.overnight 2>&1 &
stdbuf -oL nohup sh -c  'sh run_all.sh ALL --background --nogpu' > out.overnight 2>&1 &
