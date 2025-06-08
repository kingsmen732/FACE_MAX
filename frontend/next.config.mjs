/** @type {import('next').NextConfig} */
const nextConfig = {
  env: {
    NEXT_PUBLIC_API_UPLOAD_IMAGE: process.env.NEXT_PUBLIC_API_UPLOAD_IMAGE,
    NEXT_PUBLIC_API_SUBMIT_ANSWERS: process.env.NEXT_PUBLIC_API_SUBMIT_ANSWERS,
    NEXT_PUBLIC_API_GET_RESULTS: process.env.NEXT_PUBLIC_API_GET_RESULTS,
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
}

export default nextConfig
