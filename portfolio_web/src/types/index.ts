// src/types/index.ts
export interface User {
    id: number;
    email: string;
    name: string;
  }
  
  export interface Project {
    id: number;
    title: string;
    description: string;
    demoPath: string;
  }
  
  export interface DemoCredits {
    [key: string]: number;
  }
  
  export interface AuthState {
    user: User | null;
    token: string | null;
    demoCredits: DemoCredits;
  }
  
  export interface LoginCredentials {
    email: string;
    password: string;
  }
  
  export interface DemoResult {
    query: string;
    result: any;
    timestamp: Date;
  }
  