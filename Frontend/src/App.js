import {Navigate,Route,Routes} from 'react-router-dom';
import Login from './pages/Login';
import Signup from './pages/Signup';
import Home from './pages/Home';
import DL from './pages/DL'
import GenAI from './pages/GenAI'
import './App.css';
import Navbar from './pages/nav';
import Vit from './pages/ViT'

function App() {
  return (
    <div className="App">
      <Navbar/>
      <Routes>
        <Route path='/' element={<Navigate to="/login"/>}/>
        <Route path='/login' element={<Login/>}/>
        <Route path='/signup' element={<Signup/>}/>
        <Route path='/home' element={<Home/>}/>
        <Route path='/DL' element={<DL/>}/>
        <Route path='/GenAI' element={<GenAI/>}/>
        <Route path='/vit' element={<Vit/>}/>
      </Routes>
    </div>
  );
}

export default App;
