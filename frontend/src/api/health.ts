import apiClient from "./client";

export interface HealthResponse {
  status: string;
}

export const checkBackendHealth = async (): Promise<HealthResponse> => {
  const response = await apiClient.get<HealthResponse>("/api/v1/health");
  return response.data;
};