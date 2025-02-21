import streamlit as st
import tempfile
import os
import asyncio
from main import classify_image, retrieve_information

# CSS
st.markdown(
    """
    <style>
        /* Global Styles */
        body {
            background-color: #111;
            color: #eee;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
        }
        
        /* Header */
        .main-header {
            font-size: 2.5rem;
            color: #ff5252;
            text-align: center;
            margin-top: 20px;
            font-weight: bold;
        }

        /* Image Upload Box */
        .upload-box {
            background-color: #222;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
            margin-bottom: 30px;
        }
        
        .btn {
            background-color: #ff5252;
            color: #fff;
            padding: 12px 25px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .btn:hover {
            background-color: #ff6f6f;
            transform: scale(1.05);
        }

        .footer {
            text-align: center;
            margin-top: 50px;
            color: #aaa;
            font-size: 0.9rem;
            padding: 20px;
            border-top: 2px solid #444;
        }

        /* Chat Box */
        .chat-box {
            max-height: 500px;
            overflow-y: auto;
            margin-bottom: 80px;
            padding: 20px;
            background-color: #1a1a1a;
            border-radius: 10px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.5);
        }

        .message {
            background-color: #333;
            padding: 12px;
            margin: 8px 0;
            border-radius: 10px;
            color: #eee;
            font-size: 14px;
            max-width: 75%;
            word-wrap: break-word;
        }

        .user-message {
            background-color: #ff5252;
            color: white;
            align-self: flex-end;
        }

        .bot-message {
            background-color: #444;
            color: white;
        }

        .input-container {
            position: fixed;
            bottom: 20px;
            left: 5%;
            right: 5%;
            display: flex;
            align-items: center;
            background-color: #222;
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        }

        .input-box {
            flex: 1;
            padding: 12px;
            border-radius: 5px;
            border: none;
            margin-right: 15px;
            background-color: #333;
            color: #eee;
            font-size: 16px;
        }

        .input-box:focus {
            outline: none;
            border: 2px solid #ff5252;
        }

        .send-btn {
            background-color: #ff5252;
            color: white;
            padding: 12px 25px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .send-btn:hover {
            background-color: #ff6f6f;
            transform: scale(1.05);
        }
        
        .input-label {
            position: absolute;
            top: 5px;
            left: 20px;
            color: #aaa;
            font-size: 14px;
        }

        /* Smooth scroll for chat */
        .chat-box::-webkit-scrollbar {
            width: 8px;
        }

        .chat-box::-webkit-scrollbar-thumb {
            background-color: #ff5252;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Title
st.markdown("<h1 class='main-header'>Brain Tumor Prediction</h1>", unsafe_allow_html=True)

# storing chat history 
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'image_uploaded' not in st.session_state:
    st.session_state.image_uploaded = False

# Function for image upload and prediction
def handle_image_upload():
    uploaded_image = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
            tmpfile.write(uploaded_image.getbuffer())
            img_path = tmpfile.name

        # Display img
        st.image(img_path, caption="Uploaded MRI Image", use_container_width=True)

        # Classify img
        st.markdown("### Result")
        with st.spinner("Processing the image..."):
            try:
                tumor_type = classify_image(img_path)
                if tumor_type == "No tumor":
                    st.success("Great news! No tumor detected 😄")
                else:
                    st.success(f"Tumor Detected: {tumor_type}")
            except Exception as e:
                st.error(f"Error in classification: {e}")

        # Assigning Button to Show "More Details"
        if tumor_type and tumor_type != "No tumor":
            if st.button("Show More Details"):
                with st.spinner("Fetching detailed information..."):
                    try:
                        tumor_info = retrieve_information(tumor_type)
                        st.markdown(f"### Details about {tumor_type}")
                        st.write(tumor_info)
                    except Exception as e:
                        st.error(f"Error retrieving information: {e}")

        
        st.session_state.image_uploaded = True

handle_image_upload()

# To process query 
async def process_query(query):
    # chat history
    st.session_state.chat_history.append({"role": "user", "content": query})

    # Calling retrieve_information 
    with st.spinner("Processing the query..."):
        try:
            response = await asyncio.to_thread(retrieve_information, query)
        except Exception as e:
            response = f"Error: {e}"

    # response to chat history
    st.session_state.chat_history.append({"role": "bot", "content": response})

    
    return response

# Input box for queries
if st.session_state.image_uploaded:
    # To Display chat history
    st.markdown("<div class='chat-box' id='chat-box'>", unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        if msg['role'] == 'user':
            st.markdown(f"<div class='message user-message'>{msg['content']}</div>", unsafe_allow_html=True)
        elif msg['role'] == 'bot':
            st.markdown(f"<div class='message bot-message'>{msg['content']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Input box
    with st.container():
        query = st.text_input("", label_visibility="hidden", placeholder="Enter your query...")

        # submits the query
        if st.button("Submit") and query:
            #  query processing
            asyncio.run(process_query(query))
            
            # To clear the box
            query = ""
            
            st.rerun()

    # Smooth scrolling 
    st.markdown("""<script>
        var chatBox = document.getElementById("chat-box");
        chatBox.scrollTop = chatBox.scrollHeight;
    </script>""", unsafe_allow_html=True)

# Footer
st.markdown(
    "<div class='footer'>© 2025 Brain Tumor Prediction App | All Rights Reserved</div>",
    unsafe_allow_html=True,
)
