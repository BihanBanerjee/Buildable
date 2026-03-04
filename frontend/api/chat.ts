import { apiClient } from "./client";
import { ChatResponse, Project } from "./types";

export const chatApi = {
  createChat: async (
    prompt: string,
    modelChoice: string = "gemini",
  ): Promise<ChatResponse> => {
    const response = await apiClient.post<ChatResponse>("/chat", {
      prompt,
      model_choice: modelChoice,
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

  checkUrlHealth: async (url: string): Promise<boolean> => {
    try {
      const response = await apiClient.head(url, { timeout: 5000 });
      return response.status === 200;
    } catch {
      return false;
    }
  },
};
