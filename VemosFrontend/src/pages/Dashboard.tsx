import NavBar from '../components/NavBar'

import '../style/Dashboard.css'
import '../../node_modules/react-vis/dist/style.css';

import Carousel from 'nuka-carousel';

import { BsFillArrowRightCircleFill, BsFillArrowLeftCircleFill  } from 'react-icons/bs';
import { IoMdAddCircle  } from 'react-icons/io';

import {
  XYPlot,
  VerticalGridLines,
  HorizontalGridLines,
  XAxis,
  YAxis,
  Crosshair,
  VerticalBarSeries,
  VerticalBarSeriesPoint,
  HorizontalBarSeries,
  HorizontalBarSeriesPoint
} from "react-vis";

import TextField from '@mui/material/TextField';
import Autocomplete from '@mui/material/Autocomplete';

import dayjs, { Dayjs } from 'dayjs';
import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { DateTimePicker } from '@mui/x-date-pickers/DateTimePicker';

import { useQuery } from 'react-query';
import axios from "axios";
import { useEffect, useState } from 'react';

interface Vehicle {
  _id: string,
  id: string,
  plate: string,
  color: string,
  model: string,
  brand: string,
  category: string,
  possibleImages: string[],
  createdAt: string
}

interface Filter {
  label: string,
  field: string
}

interface QueryFilter {
  model: string,
  brand: string,
  category: string,
  plate: string,
}

interface AggReturn {
  x: number,
  y: string
}

interface DateValue {
  "$gte": Dayjs | null,
  "$lt": Dayjs | null,
}

async function countVehicles(filter : QueryFilter | {}) {
  return axios.post("https://labic.utfpr.edu.br/vemos/backend/vehicles/count", { filter }).then((res) => res.data)
}

async function vehiclesAgg(field: string, filter : QueryFilter | {}) {
  return axios.post("https://labic.utfpr.edu.br/vemos/backend/vehicles/agg", { field, filter }).then((res) => res.data)
}

async function listOfVehicles(filter : QueryFilter | {}) {
  return axios.post("https://labic.utfpr.edu.br/vemos/backend/vehicles", { filter }).then((res) => res.data)
}

async function filters() {
  return axios.get("https://labic.utfpr.edu.br/vemos/backend/vehicles/fields").then((res) => res.data)
}

async function values(field : string | undefined, filter : QueryFilter | {}) {
  return axios.post(`https://labic.utfpr.edu.br/vemos/backend/vehicles/values`, { field, filter }).then((res) => res.data)
}

function Dashboard() {
  const [ vehicle, setVehicle ] = useState<Vehicle | undefined>(undefined);
  const [ checked, setChecked ] = useState<string | undefined>(undefined);
  const [ fieldSelected, setFieldSelected ] = useState<Filter | undefined>(undefined);
  const [ valueSelected, setValueSelected ] = useState<string | undefined>(undefined);
  const [ filterToApply, setFilterToApply ] = useState<QueryFilter | {}>({});
  const [ barValue, setBarValue ] = useState<VerticalBarSeriesPoint[]>([]);
  const [ horBarValue, setHorBarValue ] = useState<HorizontalBarSeriesPoint[]>([]);
  const [ dateValue, setDateValue ] = useState<DateValue>({ "$gte": dayjs(), "$lt":dayjs()});
  
  const { data: colorsData } = useQuery(["colorsAgg", filterToApply], () => vehiclesAgg("color", filterToApply));
  let { data: brandsData } = useQuery(["brandsAgg", filterToApply], () => vehiclesAgg("brand", filterToApply));
  const { data : vehicleCount  } = useQuery(["countVehices", filterToApply], () => countVehicles(filterToApply));
  const { data : vehiclesList  } = useQuery(["listOfVehicles", filterToApply], () => listOfVehicles(filterToApply));
  const { data : filtersList  } = useQuery(["filters"], filters);
  const { data : valuesList  } = useQuery(["values", fieldSelected, filterToApply], () => values(fieldSelected?.field, filterToApply));

  // const dataColors = colorsData ? colorsData.data : []
  // const dataBrands = brandsData ? brandsData.data : []
  // const listVehicles : Vehicle[] = vehiclesList ? vehiclesList.data : []
  // const listFilters : Filter[] = filtersList ? filtersList.data : []

  useEffect(() => {
    if(vehiclesList && vehiclesList.length > 0) {
      setVehicle(vehiclesList[0])
      setChecked(vehiclesList[0]._id)
    }
  }, [vehiclesList])


  let carousel = [<div></div>]
  if(vehicle) {
    carousel = vehicle.possibleImages.map((image: string) => (
      <img src= {`https://labic.utfpr.edu.br/vemos/data/${vehicle.id}/${image}`} alt= ""/>
    ))
  }

  function onCheckbox(event: React.ChangeEvent<HTMLInputElement>) {
    if(event.target.checked){
      const newVehicle : Vehicle | undefined = vehiclesList.find((vehicle : Vehicle) => vehicle._id === event.target.value);
      setVehicle(newVehicle)
      setChecked(newVehicle ? newVehicle._id : undefined)
    }
  }

  async function handleFieldSelection(event: React.SyntheticEvent, value : Filter | null ) {
    if(value){
      setFieldSelected(value);
    }
    else {
      setFieldSelected(undefined);
    }
  }

  async function onValueSelected(event: React.SyntheticEvent, value : string | null ) {
    if(value){
      setValueSelected(value);
    }
    else {
      setValueSelected(undefined);
    }
  }

  async function addFilter(){
    if (fieldSelected && fieldSelected.field !== "createdAt")
      setFilterToApply({ ...filterToApply, [fieldSelected.field]: valueSelected })
    else if(fieldSelected && fieldSelected.field === "createdAt")
      setFilterToApply({ ...filterToApply, [fieldSelected.field]: dateValue })
  }

  async function cleanFilter(){
    setFilterToApply({})
  }

  return (
    <div>
      <NavBar />
      <div className='container'>
        <div className='header'>
          <div className='filter'>
            <div className='autocompletes'>
              <Autocomplete
                disablePortal
                id="combo-box-demo"
                options={filtersList ? filtersList : []}
                sx={{ width: 250 }}
                onChange={handleFieldSelection}
                className="filter-bar"
                renderInput={(params) => <TextField {...params} label="Escolha o filtro..." />}
              />
              {fieldSelected && fieldSelected.label !== "Data" ? (
                <>
                  <Autocomplete
                    disablePortal
                    id="combo-box-demo"
                    options={valuesList ? valuesList : []}
                    sx={{ width: 250 }}
                    className="filter-bar"
                    onChange={onValueSelected}
                    renderInput={(params) => <TextField {...params} label={fieldSelected.label} />}
                  />
                  <button className='button-add' onClick={ addFilter }><IoMdAddCircle /></button>
                </>
              ) : fieldSelected && fieldSelected.label === "Data" ? (
                <>
                  <LocalizationProvider dateAdapter={AdapterDayjs}>
                    <DateTimePicker
                      renderInput={(props) => <TextField {...props} />}
                      label="Inicio"
                      className="filter-bar"
                      value={dateValue["$gte"]}
                      onChange={(newValue: Dayjs | null) => {
                        setDateValue({...dateValue, "$gte": newValue});
                      }}
                    />
                    <DateTimePicker
                      renderInput={(props) => <TextField {...props} />}
                      label="Fim"
                      className="filter-bar"
                      value={dateValue["$lt"]}
                      onChange={(newValue: Dayjs | null) => {
                        setDateValue({...dateValue, "$lt": newValue});
                      }}
                    />
                  </LocalizationProvider>
                  <button className='button-add' onClick={ addFilter }><IoMdAddCircle /></button>
                </>
              ):undefined}
              { Object.entries(filterToApply).map(([key, value]) => (
                key !== "createdAt" 
                  ? (<div className='filter-chip'> {value} </div>)
                  : (<>
                      <div className='filter-chip'> {value["$gte"]["$d"].toLocaleString()} </div> 
                      <div className='filter-chip'> {value["$lt"]["$d"].toLocaleString()} </div>
                    </>)
              ))}
            </div>
            <button className='button-filter' onClick={cleanFilter}>Limpar</button>
          </div>
          <hr />
        </div>
        <div className='chip item'>
          <p>Quantidade total de carros</p>
          <h2>{vehicleCount ? vehicleCount.count : 0}</h2>
        </div>
        <div className='graph-1 item'>
          <p>Quantidade de carros por cor</p>
          <XYPlot
            animation
            xType="ordinal"
            width={400}
            height={400}
            color="#852BFF"
            className={"graphs"}
            onMouseLeave={() => setBarValue([])}
          >
            <VerticalGridLines />
            <HorizontalGridLines />
            <XAxis />
            <YAxis />
            <Crosshair values={barValue}/>
            <VerticalBarSeries 
              data={colorsData}
              barWidth={0.7} 
              onValueMouseOver={(datapoint, event)=>{
                setBarValue([datapoint]);
              }}
            />
          </XYPlot>
        </div>
        <div className='graph-2 item'>
          <p>Quantidade de carros por marca</p>
          <XYPlot
            width={500}
            height={400}
            margin={{left: 120}}
            color="#852BFF"
            className={"graphs"}
            yType="ordinal"
            onMouseLeave={() => setHorBarValue([])}
          >
            <VerticalGridLines/>
            <HorizontalGridLines />
            <XAxis/>
            <YAxis />
            <Crosshair values={horBarValue}/>
            <HorizontalBarSeries 
              data={brandsData?.map((data : AggReturn) => ({x: data.y ,y: data.x}))} 
              barWidth={0.5} 
              onValueMouseOver={(datapoint, event)=>{
                setHorBarValue([datapoint]);
              }}
            />
          </XYPlot>
        </div>
        <div className='list item'>
          <p>Lista de carros</p>
          {vehiclesList?.map((vehicle: Vehicle) => (
            <div id={vehicle._id}>
              <div className='li'>
                <div>
                <label className="check">
                  <input type="checkbox" onChange={onCheckbox} value={vehicle._id} checked={vehicle._id === checked}/>
                  <span className="checkmark"></span>
                </label>
                  <h2>{vehicle.category} - {vehicle.brand} {vehicle.plate}</h2>
                </div>
                <div>
                  <p>Cor: {vehicle.color} / Modelo: {vehicle.model}</p>
                </div>
              </div>
              <hr />
            </div>
          ))}
        </div>
        <div className='info item'>
          <p>Detalhes</p>
          {vehicle ? (
            <div className='vehicle-container'>
              <div className='vehicle-info'>
                <div className='vehicle-item'>
                  <div className='vehicle-chip'>
                    <p>Categoria</p>
                    <h1>{vehicle.category}</h1>
                  </div>
                  <div className='vehicle-chip'>
                    <p>Marca</p>
                    <h1>{vehicle.brand}</h1>
                  </div>
                  <div className='vehicle-chip'>
                    <p>Modelo</p>
                    <h1>{vehicle.model}</h1>
                  </div>
                </div>
                <hr />
                <div className='vehicle-item'>
                  <div className='vehicle-chip'>
                    <p>Cor</p>
                    <h1>{vehicle.color}</h1>
                  </div>
                  <div className='vehicle-chip'>
                    <p>Placa</p>
                    <h1>{vehicle.plate}</h1>
                  </div>
                </div>
                <hr />
                <div className='vehicle-item'>
                  <div className='vehicle-chip data'>
                    <p>Visto por último</p>
                    <h1>{vehicle.createdAt}</h1>
                  </div>
                </div>
              </div>
              <div className='carousel'>
                  <Carousel 
                    renderCenterLeftControls={({ previousSlide }) => (
                      <button id='button' onClick={previousSlide}> <BsFillArrowLeftCircleFill /> </button>
                    )}
                    renderCenterRightControls={({ nextSlide }) => (
                      <button id='button' onClick={nextSlide}> <BsFillArrowRightCircleFill/> </button>
                    )}
                  >
                    {carousel}
                  </Carousel>
                </div>
            </div>
          ) : undefined}
        </div>
      </div>
    </div>
  )
}
  
export default Dashboard;
  