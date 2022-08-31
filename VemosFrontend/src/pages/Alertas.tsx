import NavBar from "../components/NavBar"

import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';

import Dialog from '@mui/material/Dialog';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';
import DialogTitle from '@mui/material/DialogTitle';

import { Formik, Field, Form } from 'formik';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { useState } from "react";
import axios from "axios";

import { ToastContainer, toast } from 'react-toastify';
import { AiFillDelete  } from 'react-icons/ai';

import 'react-toastify/dist/ReactToastify.css';
import '../style/Alertas.css'

interface Alerta {
  _id?: string,
  email?: string,
  model?: string,
  brand?: string,
  category?: string,
  color?: string,
  plate?: string
}


async function listOfAlertas() {
  return axios.get("https://labic.utfpr.edu.br/vemos/backend/alertas").then((res) => res.data)
}

function Alertas() {
  const [open, setOpen] = useState(false);
  const queryClient = useQueryClient()

  const { data : alertasList  } = useQuery(["listOfAlertas"], listOfAlertas);

  const addAlerta = (alerta : Alerta) => {
    return axios.post("https://labic.utfpr.edu.br/vemos/backend/alertas", { alerta })
  }

  const mutation = useMutation(addAlerta, {
    onSuccess: () => {
      // Invalidate and refetch
      queryClient.invalidateQueries(['listOfAlertas'])
    },
  })

  const deleteAlerta = (_id : string | undefined) => {
    return axios.post("https://labic.utfpr.edu.br/vemos/backend/alertas/delete", { _id })
  }

  const mutationDeleteAlerta = useMutation(deleteAlerta, {
    onSuccess: () => {
      // Invalidate and refetch
      queryClient.invalidateQueries(['listOfAlertas'])
    },
  })

  const handleClickOpen = () => {
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
  };

  return (
    <div>
      <NavBar />
      <div className="container-alertas">
        <div className='header-alertas'>
          <button className="add-alerta" type="button" onClick={handleClickOpen} >Novo Alerta</button>
          <Dialog open={open} onClose={handleClose}>
            <DialogTitle>Criar Novo Alerta</DialogTitle>
            <DialogContent>
              <DialogContentText>
                Para criar um novo alerta preencha as informações abaixo. Quando algum veiculo com os dados preenchidos
                for encontrado, iremos enviar um email para você.
              </DialogContentText>
              <Formik
                initialValues={{
                  email: '',
                  model: '',
                  brand: '',
                  category: '',
                  color: '',
                }}
                onSubmit={async (values) => {
                  console.log(values)
                  mutation.mutate(values)
                  toast("Adicionando alerta!!!", { position: toast.POSITION.BOTTOM_RIGHT });
                  handleClose();
                }}
              >
                <Form className="forms">
                  <Field
                    id="email"
                    name="email"
                    placeholder="E-mail"
                    type="email"
                  />

                  <Field id="category" name="category" placeholder="Categoria" />

                  <Field id="brand" name="brand" placeholder="Marca" />

                  <Field id="model" name="model" placeholder="Modelo" />

                  <Field id="color" name="color" placeholder="Cor" />
                  <br />
                  <button type="submit">Criar alerta</button>
                </Form>
              </Formik>
            </DialogContent>
          </Dialog>
        </div>
        <hr />
        <div className="tabela-container">
          <TableContainer component={Paper} sx={{ maxWidth: "96vw" }}>
            <Table sx={{ minWidth: 500 }} aria-label="simple table">
              <TableHead>
                <TableRow>
                  <TableCell>E-mail</TableCell>
                  <TableCell align="center">Categoria</TableCell>
                  <TableCell align="center">Marca</TableCell>
                  <TableCell align="center">Modelo</TableCell>
                  <TableCell align="center">Placa</TableCell>
                  <TableCell align="center">Cor</TableCell>
                  <TableCell align="center"></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {alertasList?.map((alerta : Alerta ) => (
                  <TableRow
                    key={alerta._id}
                    sx={{ '&:last-child td, &:last-child th': { border: 0 } }}
                  >
                    <TableCell component="th" scope="alerta">
                      {alerta.email}
                    </TableCell>
                    <TableCell align="center">{alerta.category ? alerta.category : "-----"}</TableCell>
                    <TableCell align="center">{alerta.brand ? alerta.brand : "-----"}</TableCell>
                    <TableCell align="center">{alerta.model ? alerta.model : "-----"}</TableCell>
                    <TableCell align="center">{alerta.plate ? alerta.plate : "-----"}</TableCell>
                    <TableCell align="center">{alerta.color ? alerta.color : "-----"}</TableCell>
                    <TableCell align="center"><AiFillDelete className="delete-button" onClick={() => mutationDeleteAlerta.mutate(alerta._id)} /></TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </div>
        <ToastContainer />
      </div>
    </div>
  )
  }
  
export default Alertas;
  