import {
  BrowserRouter,
  Routes,
  Route,
  Navigate
} from "react-router-dom";


import Login  from "./pages/Login";
import Alertas from "./pages/Alertas";
import Dashboard from "./pages/Dashboard";
import { useEffect, useState } from "react";


export function AppRoutes(){
  const [isAuthenticated, setIsAuthenticated] = useState(localStorage.getItem("token"))

  useEffect(()=> {
    setIsAuthenticated(localStorage.getItem("token"))
    console.log(localStorage.getItem("token"))
  },[])

  return(
  <BrowserRouter basename="/vemos/app">
    <Routes>
      <Route path="/" element={<Login setAuthenticate={setIsAuthenticated} />} />
      <Route path="/dashboard" element={isAuthenticated ? <Dashboard /> : <Navigate to="/" /> }/>
      <Route path="/alertas" element={isAuthenticated ? <Alertas /> : <Navigate to="/" />}/>
    </Routes>
  </BrowserRouter>)
};