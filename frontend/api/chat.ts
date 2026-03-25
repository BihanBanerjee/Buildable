import { apiClient } from "./client";
import { ChatResponse, Project } from "./types";

export const chatApi = {
  createChat: async (
    prompt: string,
  ): Promise<ChatResponse> => {
    const response = await apiClient.post<ChatResponse>("/chat", {
      prompt,
    });
    return response.data;
  },

  listProjects: async (): Promise<{ projects: Project[] }> => {
    const response = await apiClient.get<{ projects: Project[] }>("projects");
    return response.data;
  },

  deleteProject: async (projectId: string): Promise<void> => {
    await apiClient.delete(`/projects/${projectId}`);
  },

  deployProject: async (
    projectId: string,
  ): Promise<{ success: boolean; url: string }> => {
    const response = await apiClient.post<{ success: boolean; url: string }>(
      `/projects/${projectId}/deploy`,
    );
    return response.data;
  },

  checkUrlHealth: async (url: string): Promise<boolean> => {
    try {
      const response = await apiClient.head(url, { timeout: 5000 });
      return response.status === 200;
    } catch {
      return false;
    }
  },
};
