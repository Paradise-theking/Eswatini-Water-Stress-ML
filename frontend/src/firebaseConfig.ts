// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";
import { getAnalytics } from "firebase/analytics";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyCEQnvzD2zWSNj0kIiXOFrx9A7DizyuKNY",
  authDomain: "eswatiniwaterstress.firebaseapp.com",
  projectId: "eswatiniwaterstress",
  storageBucket: "eswatiniwaterstress.firebasestorage.app",
  messagingSenderId: "1088599063436",
  appId: "1:1088599063436:web:48772bfb0a2f8f752db4a9",
  measurementId: "G-Q4XXZVZJHW"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);

// CRITICAL: Ensure 'export' is added here so main.ts can access it!
export const db = getFirestore(app);