import axios from 'axios';
import toast from 'react-hot-toast';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      const message = error.response.data?.detail || 'An error occurred';
      toast.error(message);
    } else if (error.request) {
      toast.error('Network error. Please check your connection.');
    }
    return Promise.reject(error);
  }
);

export const API = {
  health: () => apiClient.get('/health'),
  stats: () => apiClient.get('/stats'),
  query: (data) => apiClient.post('/query', data),
  classifyFish: (formData) => apiClient.post('/classify/fish', formData, { headers: { 'Content-Type': 'multipart/form-data' } }),
  detectDisease: (formData) => apiClient.post('/detect/disease', formData, { headers: { 'Content-Type': 'multipart/form-data' } }),
};

export default apiClient;
