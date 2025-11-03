import { supabase } from './client'

export interface SignUpData {
  email: string
  password: string
  name?: string
}

export interface SignInData {
  email: string
  password: string
}

/**
 * Sign up a new user
 */
export async function signUp({ email, password, name }: SignUpData) {
  try {
    const { data, error } = await supabase.auth.signUp({
      email,
      password,
      options: {
        data: {
          name: name || email.split('@')[0],
        },
      },
    })

    if (error) throw error

    return { data, error: null }
  } catch (error: any) {
    return { data: null, error: error.message }
  }
}

/**
 * Sign in an existing user
 */
export async function signIn({ email, password }: SignInData) {
  try {
    const { data, error } = await supabase.auth.signInWithPassword({
      email,
      password,
    })

    if (error) throw error

    return { data, error: null }
  } catch (error: any) {
    return { data: null, error: error.message }
  }
}

/**
 * Sign out the current user
 */
export async function signOut() {
  try {
    const { error } = await supabase.auth.signOut()
    if (error) throw error
    return { error: null }
  } catch (error: any) {
    return { error: error.message }
  }
}

/**
 * Get the current user session
 */
export async function getSession() {
  try {
    const { data: { session }, error } = await supabase.auth.getSession()
    if (error) throw error
    return { session, error: null }
  } catch (error: any) {
    return { session: null, error: error.message }
  }
}

/**
 * Get the current user
 */
export async function getUser() {
  try {
    const { data: { user }, error } = await supabase.auth.getUser()
    if (error) throw error
    return { user, error: null }
  } catch (error: any) {
    return { user: null, error: error.message }
  }
}

/**
 * Reset password request
 */
export async function resetPassword(email: string) {
  try {
    const { error } = await supabase.auth.resetPasswordForEmail(email, {
      redirectTo: `${process.env.NEXT_PUBLIC_APP_URL}/auth/reset-password`,
    })
    if (error) throw error
    return { error: null }
  } catch (error: any) {
    return { error: error.message }
  }
}

/**
 * Update user password
 */
export async function updatePassword(newPassword: string) {
  try {
    const { error } = await supabase.auth.updateUser({
      password: newPassword,
    })
    if (error) throw error
    return { error: null }
  } catch (error: any) {
    return { error: error.message }
  }
}
