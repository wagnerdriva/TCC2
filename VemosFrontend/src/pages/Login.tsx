import logo from '../assets/vemos_color.png'
import ilustra from '../assets/ilustra.png'

import '../style/Login.css'

import axios from "axios";
import React, { useState } from 'react';

import { ToastContainer, toast } from 'react-toastify';
import { useNavigate } from "react-router-dom";

interface User {
  cn: string,
  login: string,
  token: string
}

function Login(props: any) {
  const navigate = useNavigate();
  const [ login, setLogin ] = useState("");
  const [ password, setPassword ] = useState("");

  function handleChangtLogin(event: React.ChangeEvent<HTMLInputElement>){
    setLogin(event.target.value)
  }

  function handleChangePassword(event: React.ChangeEvent<HTMLInputElement>){
    setPassword(event.target.value)
  }

  async function autenthication() : Promise<User>{
    return new Promise((resolve, reject) => {
      axios.post("https://labic.utfpr.edu.br/vemos/backend/authenticate", { login, password })
        .then(response =>{ 
          if(response.data.error === "Invalid Credentials")
            reject(response.data.error)
          else{
            resolve(response.data)
          }
        })
        .catch(error => reject(error))
    });
  }

  async function executeAuthentication() {
    try{
      const user : User = await autenthication();
      console.log(user)
      localStorage.setItem("token", user.token);
      props.setAuthenticate(true)
      if(user.token){
        navigate("/dashboard", { replace: true })
      }
    }
    catch(error){
      console.log(error)
      toast.error("Falha no login!!!", { position: toast.POSITION.BOTTOM_RIGHT });
    }
    
  }

  return  (
    <div className='login-container'>
      <div className='ilustracao'>
        <img src={ilustra} alt="ilustracao"/>
      </div>
      <div className='login'>
        <img src={logo} alt="logo" id='logo'/>
        <h3>Entre na sua conta</h3>
        <input 
          className="login-input" 
          placeholder='E-mail' 
          type='text'
          onChange={handleChangtLogin} 
          value={login} />
        <input 
          className="login-input" 
          placeholder='Senha' 
          type='password' 
          onChange={handleChangePassword} 
          value={password} />
        <button className='button' type='button' onClick={executeAuthentication}>Entrar</button>
      </div>
      <ToastContainer />
    </div>
  )
}

export default Login;
