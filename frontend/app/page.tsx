"use client"

import type React from "react"

import { useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Camera, CheckIcon as Assessment, TrendingUp, CheckCircle, CloudUpload } from "lucide-react"
import Image from "next/image"

interface FormData {
  sleepHours: string
  skincare: string
  workout: string
  waterIntake: string
  processedFoods: string
}

interface Results {
  currentScore: number
  potentialScore: number
}

export default function FaceMaxingApp() {
  const [currentStep, setCurrentStep] = useState<"upload" | "questions" | "results">("upload")
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const [formData, setFormData] = useState<FormData>({
    sleepHours: "",
    skincare: "",
    workout: "",
    waterIntake: "",
    processedFoods: "",
  })
  const [results, setResults] = useState<Results | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [uploadSuccess, setUploadSuccess] = useState(false)

  // API endpoints from environment variables with fallbacks
  const API_UPLOAD_IMAGE = process.env.NEXT_PUBLIC_API_UPLOAD_IMAGE || "https://face-max.onrender.com/upload-image"
  const API_SUBMIT_ANSWERS = process.env.NEXT_PUBLIC_API_SUBMIT_ANSWERS || "https://face-max.onrender.com/submit-answers"
  const API_GET_RESULTS = process.env.NEXT_PUBLIC_API_GET_RESULTS || "https://face-max.onrender.com/get-results"

  const handleImageUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    setIsLoading(true)
    setError(null)

    try {
      // Create preview
      const reader = new FileReader()
      reader.onload = (e) => {
        setUploadedImage(e.target?.result as string)
      }
      reader.readAsDataURL(file)

      // Upload to backend
      const uploadFormData = new FormData()
      uploadFormData.append("image", file)

      const response = await fetch(API_UPLOAD_IMAGE, {
        method: "POST",
        body: uploadFormData,
      })

      if (!response.ok) {
        const errorData = await response.text()
        throw new Error(`Upload failed: ${response.status} - ${errorData}`)
      }

      const result = await response.json()
      console.log("Upload successful:", result)

      setUploadSuccess(true)
      setTimeout(() => setUploadSuccess(false), 3000)

      setTimeout(() => {
        setCurrentStep("questions")
        setIsLoading(false)
      }, 1000)
    } catch (err) {
      console.error("Upload error:", err)
      setError(`Failed to upload image: ${err instanceof Error ? err.message : "Unknown error"}`)
      setIsLoading(false)
    }
  }

  const handleSubmit = async () => {
    setIsLoading(true)
    setError(null)

    try {
      // Submit answers
      const response = await fetch(API_SUBMIT_ANSWERS, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          ...formData,
          sleepHours: Number.parseFloat(formData.sleepHours),
          waterIntake: Number.parseFloat(formData.waterIntake),
        }),
      })

      if (!response.ok) {
        const errorData = await response.text()
        throw new Error(`Submission failed: ${response.status} - ${errorData}`)
      }

      // Get results
      const resultsResponse = await fetch(API_GET_RESULTS)
      if (!resultsResponse.ok) {
        const errorData = await resultsResponse.text()
        throw new Error(`Results fetch failed: ${resultsResponse.status} - ${errorData}`)
      }

      const resultsData = await resultsResponse.json()
      console.log("Results received:", resultsData)

      // Simulate processing time for better UX
      setTimeout(() => {
        setResults(resultsData)
        setCurrentStep("results")
        setIsLoading(false)
      }, 1500)
    } catch (err) {
      console.error("Submission error:", err)
      setError(`Failed to process your data: ${err instanceof Error ? err.message : "Unknown error"}`)
      setIsLoading(false)
    }
  }

  const resetApp = () => {
    setCurrentStep("upload")
    setUploadedImage(null)
    setFormData({
      sleepHours: "",
      skincare: "",
      workout: "",
      waterIntake: "",
      processedFoods: "",
    })
    setResults(null)
    setError(null)
  }

  const isFormValid = () => {
    return (
      Object.values(formData).every((value) => value !== "") &&
      Number.parseFloat(formData.sleepHours) > 0 &&
      Number.parseFloat(formData.waterIntake) > 0
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* App Bar */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
              <Assessment className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-semibold text-gray-900">Face Maxing Tool</h1>
              <p className="text-sm text-gray-600">AI-powered attractiveness analysis</p>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-4 py-8">
        {/* Stepper */}
        <div className="mb-8">
          <div className="flex items-center justify-between max-w-md mx-auto">
            <div className="flex flex-col items-center">
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-medium transition-all duration-300 ${
                  currentStep === "upload"
                    ? "bg-blue-600 text-white"
                    : currentStep === "questions" || currentStep === "results"
                      ? "bg-green-500 text-white"
                      : "bg-gray-300 text-gray-600"
                }`}
              >
                {currentStep === "questions" || currentStep === "results" ? <CheckCircle className="w-5 h-5" /> : "1"}
              </div>
              <span className="text-xs text-gray-600 mt-1">Upload</span>
            </div>
            <div
              className={`flex-1 h-0.5 mx-4 transition-colors duration-300 ${
                currentStep === "questions" || currentStep === "results" ? "bg-green-500" : "bg-gray-300"
              }`}
            />
            <div className="flex flex-col items-center">
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-medium transition-all duration-300 ${
                  currentStep === "questions"
                    ? "bg-blue-600 text-white"
                    : currentStep === "results"
                      ? "bg-green-500 text-white"
                      : "bg-gray-300 text-gray-600"
                }`}
              >
                {currentStep === "results" ? <CheckCircle className="w-5 h-5" /> : "2"}
              </div>
              <span className="text-xs text-gray-600 mt-1">Questions</span>
            </div>
            <div
              className={`flex-1 h-0.5 mx-4 transition-colors duration-300 ${
                currentStep === "results" ? "bg-green-500" : "bg-gray-300"
              }`}
            />
            <div className="flex flex-col items-center">
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-medium transition-all duration-300 ${
                  currentStep === "results" ? "bg-blue-600 text-white" : "bg-gray-300 text-gray-600"
                }`}
              >
                3
              </div>
              <span className="text-xs text-gray-600 mt-1">Results</span>
            </div>
          </div>
        </div>

        {/* Error Snackbar */}
        {error && (
          <div className="mb-6 max-w-2xl mx-auto">
            <Card className="border-red-200 bg-red-50">
              <CardContent className="p-4">
                <div className="flex items-center gap-3">
                  <div className="w-6 h-6 bg-red-100 rounded-full flex items-center justify-center">
                    <span className="text-red-600 text-sm">!</span>
                  </div>
                  <p className="text-red-700">{error}</p>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {uploadSuccess && (
          <div className="mb-6 max-w-2xl mx-auto">
            <Card className="border-green-200 bg-green-50">
              <CardContent className="p-4">
                <div className="flex items-center gap-3">
                  <CheckCircle className="w-6 h-6 text-green-600" />
                  <p className="text-green-700">Image uploaded successfully!</p>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Upload Step */}
        {currentStep === "upload" && (
          <div className="max-w-2xl mx-auto">
            <Card className="shadow-sm">
              <CardContent className="p-8">
                <div className="text-center">
                  <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-6">
                    <Camera className="w-8 h-8 text-blue-600" />
                  </div>
                  <h2 className="text-2xl font-semibold text-gray-900 mb-2">Upload Your Photo</h2>
                  <p className="text-gray-600 mb-8">Upload a clear front-facing photo for accurate analysis</p>

                  {uploadedImage ? (
                    <div className="space-y-6">
                      <div className="relative w-48 h-48 mx-auto rounded-2xl overflow-hidden shadow-lg">
                        <Image
                          src={uploadedImage || "/placeholder.svg"}
                          alt="Uploaded photo"
                          fill
                          className="object-cover"
                        />
                      </div>
                      <Button
                        onClick={() => setCurrentStep("questions")}
                        className="w-full bg-blue-600 hover:bg-blue-700 text-white py-3 rounded-lg transition-colors duration-200"
                        disabled={isLoading}
                      >
                        {isLoading ? (
                          <div className="flex items-center gap-2">
                            <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                            Processing...
                          </div>
                        ) : (
                          "Continue to Questions"
                        )}
                      </Button>
                    </div>
                  ) : (
                    <div className="space-y-6">
                      <div className="border-2 border-dashed border-gray-300 rounded-2xl p-12 hover:border-blue-400 hover:bg-blue-50 transition-all duration-200">
                        <CloudUpload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                        <Label htmlFor="image-upload" className="cursor-pointer">
                          <span className="text-blue-600 font-medium hover:text-blue-700">Click to upload</span>
                          <span className="text-gray-500"> or drag and drop</span>
                          <br />
                          <span className="text-xs text-gray-400">PNG, JPG up to 10MB</span>
                        </Label>
                        <Input
                          id="image-upload"
                          type="file"
                          accept="image/*"
                          onChange={handleImageUpload}
                          className="hidden"
                        />
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Questions Step */}
        {currentStep === "questions" && (
          <div className="max-w-2xl mx-auto">
            <Card className="shadow-sm">
              <CardContent className="p-8">
                <div className="text-center mb-8">
                  <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <Assessment className="w-8 h-8 text-green-600" />
                  </div>
                  <h2 className="text-2xl font-semibold text-gray-900 mb-2">Lifestyle Assessment</h2>
                  <p className="text-gray-600">Answer these questions for personalized analysis</p>
                </div>

                <div className="space-y-8">
                  {/* Sleep Hours */}
                  <div className="space-y-3">
                    <Label className="text-base font-medium text-gray-900">
                      How many hours do you sleep per night?
                    </Label>
                    <Input
                      type="number"
                      min="1"
                      max="24"
                      step="0.5"
                      placeholder="e.g., 7.5"
                      value={formData.sleepHours}
                      onChange={(e) => setFormData((prev) => ({ ...prev, sleepHours: e.target.value }))}
                      className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>

                  {/* Skincare */}
                  <div className="space-y-3">
                    <Label className="text-base font-medium text-gray-900">Do you follow a skincare routine?</Label>
                    <RadioGroup
                      value={formData.skincare}
                      onValueChange={(value) => setFormData((prev) => ({ ...prev, skincare: value }))}
                      className="grid grid-cols-2 gap-3"
                    >
                      {["Yes", "No"].map((option) => (
                        <Label
                          key={option}
                          htmlFor={`skincare-${option}`}
                          className="flex items-center space-x-3 p-4 rounded-lg border border-gray-200 cursor-pointer hover:bg-gray-50 transition-colors duration-200 [&:has(:checked)]:bg-blue-50 [&:has(:checked)]:border-blue-300"
                        >
                          <RadioGroupItem value={option} id={`skincare-${option}`} />
                          <span className="font-medium">{option}</span>
                        </Label>
                      ))}
                    </RadioGroup>
                  </div>

                  {/* Workout */}
                  <div className="space-y-3">
                    <Label className="text-base font-medium text-gray-900">How often do you work out?</Label>
                    <RadioGroup
                      value={formData.workout}
                      onValueChange={(value) => setFormData((prev) => ({ ...prev, workout: value }))}
                      className="grid grid-cols-2 gap-3"
                    >
                      {["None", "1-2 times/week", "3-5 times/week", "6+ times/week"].map((option) => (
                        <Label
                          key={option}
                          htmlFor={`workout-${option}`}
                          className="flex items-center space-x-3 p-4 rounded-lg border border-gray-200 cursor-pointer hover:bg-gray-50 transition-colors duration-200 [&:has(:checked)]:bg-blue-50 [&:has(:checked)]:border-blue-300"
                        >
                          <RadioGroupItem value={option} id={`workout-${option}`} />
                          <span className="font-medium">{option}</span>
                        </Label>
                      ))}
                    </RadioGroup>
                  </div>

                  {/* Water Intake */}
                  <div className="space-y-3">
                    <Label className="text-base font-medium text-gray-900">Daily water intake (liters)</Label>
                    <Input
                      type="number"
                      min="0"
                      max="10"
                      step="0.1"
                      placeholder="e.g., 2.5"
                      value={formData.waterIntake}
                      onChange={(e) => setFormData((prev) => ({ ...prev, waterIntake: e.target.value }))}
                      className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>

                  {/* Processed Foods */}
                  <div className="space-y-3">
                    <Label className="text-base font-medium text-gray-900">
                      Do you frequently eat processed foods?
                    </Label>
                    <RadioGroup
                      value={formData.processedFoods}
                      onValueChange={(value) => setFormData((prev) => ({ ...prev, processedFoods: value }))}
                      className="grid grid-cols-2 gap-3"
                    >
                      {["Yes", "No"].map((option) => (
                        <Label
                          key={option}
                          htmlFor={`processed-${option}`}
                          className="flex items-center space-x-3 p-4 rounded-lg border border-gray-200 cursor-pointer hover:bg-gray-50 transition-colors duration-200 [&:has(:checked)]:bg-blue-50 [&:has(:checked)]:border-blue-300"
                        >
                          <RadioGroupItem value={option} id={`processed-${option}`} />
                          <span className="font-medium">{option}</span>
                        </Label>
                      ))}
                    </RadioGroup>
                  </div>

                  <Button
                    onClick={handleSubmit}
                    disabled={isLoading || !isFormValid()}
                    className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white py-3 rounded-lg transition-colors duration-200"
                  >
                    {isLoading ? (
                      <div className="flex items-center gap-2">
                        <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                        Analyzing...
                      </div>
                    ) : (
                      "Get My Results"
                    )}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Results Step */}
        {currentStep === "results" && results && (
          <div className="max-w-4xl mx-auto space-y-6">
            <div className="text-center mb-8">
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <TrendingUp className="w-8 h-8 text-green-600" />
              </div>
              <h2 className="text-2xl font-semibold text-gray-900 mb-2">Your Analysis Results</h2>
              <p className="text-gray-600">Based on your photo and lifestyle assessment</p>
            </div>

            {/* Score Cards */}
            <div className="grid md:grid-cols-2 gap-6">
              {/* Current Score */}
              <Card className="shadow-sm animate-in slide-in-from-left duration-500">
                <CardContent className="p-8 text-center">
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">Current Score</h3>
                  <p className="text-gray-600 text-sm mb-6">Based on current lifestyle</p>

                  <div className="relative w-32 h-32 mx-auto mb-6">
                    <svg className="w-32 h-32 transform -rotate-90" viewBox="0 0 120 120">
                      <circle cx="60" cy="60" r="50" stroke="#e5e7eb" strokeWidth="8" fill="none" />
                      <circle
                        cx="60"
                        cy="60"
                        r="50"
                        stroke="#3b82f6"
                        strokeWidth="8"
                        fill="none"
                        strokeLinecap="round"
                        strokeDasharray={`${(results.currentScore / 10) * 314} 314`}
                        className="transition-all duration-1000 ease-out"
                        style={{
                          animation: "drawCircle 1s ease-out forwards",
                        }}
                      />
                    </svg>
                    <div className="absolute inset-0 flex items-center justify-center">
                      <span className="text-3xl font-bold text-gray-900">{results.currentScore.toFixed(1)}</span>
                    </div>
                  </div>
                  <p className="text-gray-600">out of 10</p>
                </CardContent>
              </Card>

              {/* Potential Score */}
              <Card className="shadow-sm animate-in slide-in-from-right duration-500 delay-300">
                <CardContent className="p-8 text-center">
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">Potential Score</h3>
                  <p className="text-gray-600 text-sm mb-6">With optimized habits</p>

                  <div className="relative w-32 h-32 mx-auto mb-6">
                    <svg className="w-32 h-32 transform -rotate-90" viewBox="0 0 120 120">
                      <circle cx="60" cy="60" r="50" stroke="#e5e7eb" strokeWidth="8" fill="none" />
                      <circle
                        cx="60"
                        cy="60"
                        r="50"
                        stroke="#10b981"
                        strokeWidth="8"
                        fill="none"
                        strokeLinecap="round"
                        strokeDasharray={`${(results.potentialScore / 10) * 314} 314`}
                        className="transition-all duration-1000 ease-out"
                        style={{
                          animation: "drawCircle 1s ease-out 0.5s forwards",
                          strokeDasharray: "0 314",
                        }}
                      />
                    </svg>
                    <div className="absolute inset-0 flex items-center justify-center">
                      <span className="text-3xl font-bold text-gray-900">{results.potentialScore.toFixed(1)}</span>
                    </div>
                  </div>
                  <p className="text-gray-600">out of 10</p>
                </CardContent>
              </Card>
            </div>

            {/* Improvement Potential */}
            {results.potentialScore > results.currentScore && (
              <Card className="shadow-sm bg-green-50 border-green-200 animate-in fade-in duration-500 delay-1000">
                <CardContent className="p-6 text-center">
                  <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <TrendingUp className="w-6 h-6 text-green-600" />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">Improvement Potential</h3>
                  <p className="text-gray-700">
                    You could increase your score by{" "}
                    <span className="font-bold text-green-600">
                      {(results.potentialScore - results.currentScore).toFixed(1)} points
                    </span>{" "}
                    by optimizing your lifestyle habits.
                  </p>
                </CardContent>
              </Card>
            )}

            <div className="flex gap-4 justify-center">
              <Button
                onClick={resetApp}
                variant="outline"
                className="px-6 py-2 border-gray-300 text-gray-700 hover:bg-gray-50 rounded-lg transition-colors duration-200"
              >
                Start Over
              </Button>
              <Button className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors duration-200">
                Share Results
              </Button>
            </div>
          </div>
        )}
      </div>

      <style jsx>{`
        @keyframes drawCircle {
          to {
            stroke-dasharray: ${results ? (results.currentScore / 10) * 314 : 0} 314;
          }
        }
      `}</style>
    </div>
  )
}
