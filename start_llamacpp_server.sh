# Check if the file exists
if [ ! -f "models/llama-2-13b-chat.Q4_0.gguf" ]; then
    # Download the model
    cd models 
    wget -nc https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf
    cd ..
fi

# if [ ! -f "models/llama-2-7b-chat.Q4_0.gguf" ]; then
#     # Download the model
#     cd models 
#     wget -nc https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_0.gguf
#     cd ..
# fi

# Start the server
cd llama.cpp
./server -m /Users/saurabh/Documents/projects/Hand-Pose-Estimation/models/llama-2-13b-chat.Q4_0.gguf -ngl 999 -c 2048

# cd llama.cpp
# ./server -m models/llama-2-7b-chat.Q4_0.gguf -ngl 999 -c 2048
# cd ..
