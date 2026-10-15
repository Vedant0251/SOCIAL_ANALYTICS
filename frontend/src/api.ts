// Central API client — proxied through Vite to http://localhost:8000
import axios from 'axios';

const api = axios.create({ baseURL: '/api', timeout: 90000 });

export default api;

export const endpoints = {
  health:           () => api.get('/health'),
  hardware:         () => api.get('/hardware'),

  // Overview
  overviewKpis:     (ds = 'airline') => api.get(`/overview/kpis?ds=${ds}`),
  correlations:     () => api.get('/overview/correlations'),
  outliers:         () => api.get('/overview/outliers'),

  // Competitors
  competitors:      () => api.get('/competitors'),
  compTable:        () => api.get('/competitors/significance-table'),

  // Features
  features:         (ds = 'airline') => api.get(`/features?ds=${ds}`),
  featuresUpload:   (fd: FormData)   => api.post('/features/upload', fd),

  // Sentiment / ABSA
  absaMatrix:       () => api.get('/sentiment/absa-matrix'),
  sentMetrics:      () => api.get('/sentiment/metrics'),
  classReport:      () => api.get('/sentiment/classification-report'),
  absaEval:         () => api.get('/sentiment/absa-eval'),
  predict:          (text: string)   => api.post('/sentiment/predict', { text }),
  absaPredict:      (text: string)   => api.post('/sentiment/absa',    { text }),

  // Geospatial
  geoStats:         () => api.get('/geospatial/stats'),
  geoPoints:        (exact = false, limit = 2000) =>
                      api.get(`/geospatial/points?exact_only=${exact}&limit=${limit}`),

  // Network
  netSummary:       () => api.get('/network/summary'),
  netInfluencers:   (limit = 50) => api.get(`/network/influencers?limit=${limit}`),
  graphStructures:  () => api.get('/network/graph-structures'),

  // Scrapers
  bs4Quotes:        () => api.get('/scrapers/bs4-quotes'),
  sandboxQuotes:    () => api.get('/scrapers/sandbox-quotes'),
  hnFeed:           () => api.get('/scrapers/hackernews'),

  // Strategy
  decisionMatrix:   () => api.get('/strategy/decision-matrix'),
  reportMd:         () => api.get('/strategy/report-md'),
  dataGaps:         () => api.get('/strategy/data-gaps'),
};
