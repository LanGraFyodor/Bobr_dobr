import { create } from 'zustand';

interface UiState {
  isDarkMode: boolean;
  toggleDarkMode: () => void;
}

export const useUiStore = create<UiState>((set) => ({
  isDarkMode: false, 
  toggleDarkMode: () => set((state) => ({ isDarkMode: !state.isDarkMode })),
}));
