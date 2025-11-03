import { createClientComponentClient } from '@supabase/auth-helpers-nextjs'
import { createClient } from '@supabase/supabase-js'

// Client-side Supabase client for use in Client Components
export const createBrowserClient = () => {
  return createClientComponentClient()
}

// Standard Supabase client (can be used in both client and server with proper env vars)
export const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
)

// Type definitions for our database
export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export interface Database {
  public: {
    Tables: {
      networks: {
        Row: {
          id: string
          user_id: string
          name: string
          description: string | null
          node_count: number | null
          edge_count: number | null
          data: Json | null
          metadata: Json | null
          created_at: string
          updated_at: string
        }
        Insert: {
          id?: string
          user_id: string
          name: string
          description?: string | null
          node_count?: number | null
          edge_count?: number | null
          data?: Json | null
          metadata?: Json | null
          created_at?: string
          updated_at?: string
        }
        Update: {
          id?: string
          user_id?: string
          name?: string
          description?: string | null
          node_count?: number | null
          edge_count?: number | null
          data?: Json | null
          metadata?: Json | null
          created_at?: string
          updated_at?: string
        }
      }
      simulations: {
        Row: {
          id: string
          network_id: string
          user_id: string
          name: string
          simulation_type: string | null
          parameters: Json | null
          rounds: number
          repetitions: number
          results: Json | null
          status: string
          created_at: string
          completed_at: string | null
        }
        Insert: {
          id?: string
          network_id: string
          user_id: string
          name: string
          simulation_type?: string | null
          parameters?: Json | null
          rounds?: number
          repetitions?: number
          results?: Json | null
          status?: string
          created_at?: string
          completed_at?: string | null
        }
        Update: {
          id?: string
          network_id?: string
          user_id?: string
          name?: string
          simulation_type?: string | null
          parameters?: Json | null
          rounds?: number
          repetitions?: number
          results?: Json | null
          status?: string
          created_at?: string
          completed_at?: string | null
        }
      }
      layouts: {
        Row: {
          id: string
          network_id: string
          user_id: string
          name: string
          layout_type: string | null
          positions: Json | null
          parameters: Json | null
          created_at: string
        }
        Insert: {
          id?: string
          network_id: string
          user_id: string
          name: string
          layout_type?: string | null
          positions?: Json | null
          parameters?: Json | null
          created_at?: string
        }
        Update: {
          id?: string
          network_id?: string
          user_id?: string
          name?: string
          layout_type?: string | null
          positions?: Json | null
          parameters?: Json | null
          created_at?: string
        }
      }
    }
  }
}
