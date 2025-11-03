/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    domains: ['localhost'],
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '**.supabase.co',
      },
    ],
  },
  webpack: (config, { isServer }) => {
    // Handle WebGL and canvas issues
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
      path: false,
    };

    // Add worker-loader for Web Workers
    config.module.rules.push({
      test: /\.worker\.(js|ts)$/,
      use: { loader: 'worker-loader' },
    });

    return config;
  },
};

module.exports = nextConfig;
