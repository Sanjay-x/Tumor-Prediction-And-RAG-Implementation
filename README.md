# Brain Tumor Detection System

An advanced medical imaging system that combines deep learning and natural language processing to detect and analyze brain tumors from MRI scans, achieving 98% accuracy through fine-tuned CNN models and enhanced user interaction via RAG (Retrieval-Augmented Generation).

## Features

- Real-time MRI scan analysis with 98% accuracy
- Interactive chat interface for medical queries
- Advanced tumor classification (Glioma, Meningioma, Pituitary, No tumor)
- Detailed medical insights through RAG system
- Modern, responsive UI with Streamlit

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python, TensorFlow, SQLite
- **Models**: 
  - Fine-tuned CNN for image classification
  - SentenceTransformer (all-MiniLM-L6-v2) for text embeddings
  - Cohere API for enhanced response generation
- **Database**: SQLite, ChromaDB

## Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd brain-tumor-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
COHERE_API_KEY=your_api_key_here
```

4. Initialize the database:
```bash
python initialize_db.py
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Upload an MRI scan image
3. View the classification results
4. Use the chat interface for detailed medical information

## Project Structure

```
├── app.py                 # Streamlit frontend
├── main.py               # Core logic and model integration
├── initialize_db.py      # Database setup
├── Training/
│   └── fine_tuned_model.keras
├── deploy/
│   ├── tumor_info.db
│   └── Tumor_description.md
```

## Model Performance

- Base CNN Accuracy: 94%
- Fine-tuned Model Accuracy: 98%
- Supported Image Formats: JPG, JPEG, PNG
- Input Image Size: 224x224 pixels

## Security & Privacy

- Local data processing
- Secure API integrations
- No permanent storage of uploaded images

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - feel free to use and modify for your needs.

## Contact

For questions and support, please open an issue in the repository.
