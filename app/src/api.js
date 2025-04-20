const BASE_URL = 'http://localhost:8000'; // Change if hosted elsewhere

export const uploadFile = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${BASE_URL}/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) throw new Error('Upload failed');
  return res.json();
};

export const askQuestion = async (question) => {
  const res = await fetch(`${BASE_URL}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  });

  if (!res.ok) throw new Error('Query failed');
  return res.json();
};