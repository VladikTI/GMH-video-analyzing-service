import axios from 'axios';

const params = new URLSearchParams(window.location.search);
const url = params.get('api_base_url');

const API_URL = (url || import.meta.env.BASE_URL) + 'api';

const api = axios.create({
  baseURL: API_URL,
});

export default api;
