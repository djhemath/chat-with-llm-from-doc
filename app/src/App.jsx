import React, { useState } from 'react';
import { uploadFile, askQuestion } from './api';

function App() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(false);

  const handleUpload = async () => {
    if (!file) return alert('Select a file first');
    try {
      setLoading(true);
      const res = await uploadFile(file);
      console.log(res.message);
      setUploadSuccess(true);
    } catch (err) {
      alert(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleAsk = async () => {
    if (!question.trim()) return alert('Ask something!');
    try {
      setLoading(true);
      const res = await askQuestion(question);
      setAnswer(res.answer);
    } catch (err) {
      alert(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 600, margin: 'auto', padding: 30, fontFamily: 'sans-serif' }}>
      <h2>📄 RAG Document Q&A</h2>

      <input
        type="file"
        accept=".txt"
        onChange={(e) => setFile(e.target.files[0])}
      />
      <button onClick={handleUpload} disabled={loading}>
        Upload
      </button>

      {uploadSuccess && (
        <>
          <div style={{ marginTop: 30 }}>
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask something about the document..."
              style={{ width: '100%', padding: 10 }}
            />
            <button onClick={handleAsk} disabled={loading} style={{ marginTop: 10 }}>
              Ask
            </button>
          </div>
          {loading && <p>⏳ Thinking...</p>}
          {answer && (
            <div style={{ marginTop: 20, background: '#f0f0f0', padding: 15 }}>
              <strong>🤖 Answer:</strong>
              <p>{answer}</p>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default App;