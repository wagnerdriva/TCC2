import { NavLink } from "react-router-dom";

import '../style/Navbar.css'
import logo from '../assets/vemos_white.png'

function Navbar() {
    return (
      <nav className='navbar'>
        <img src={logo} alt="logo" id='logo-navbar'/>
        <NavLink className={({ isActive }) => isActive ? "links active" : "links" } to="/dashboard">Dashboard</NavLink>
        <NavLink className={({ isActive }) => isActive ? "links active" : "links" } to="/alertas">Meus Alertas</NavLink>
      </nav>
    )
  }
  
export default Navbar;
  