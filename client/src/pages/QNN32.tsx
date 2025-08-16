import {useState} from 'react'

export default function QNN32Page(){
  const [text,setText]=useState('')
  const [out,setOut]=useState<any>(null)
  const call = async ()=>{
    const r=await fetch('/api/qnn32/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})})
    setOut(await r.json())
  }
  return (<div>
    <h2>QNN32 Predictor</h2>
    <textarea value={text} onChange={e=>setText(e.target.value)} placeholder='Pega texto o señales...' rows={6} style={{width:'100%'}}/>
    <button onClick={call}>Predecir</button>
    <pre>{out?JSON.stringify(out,null,2):''}</pre>
  </div>)
}
