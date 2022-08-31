import { AppRoutes } from "./Routes";
import './style/App.css'
import { QueryClient, QueryClientProvider } from "react-query";

const queryClient = new QueryClient();

function App() {
    return (
        <QueryClientProvider client={queryClient}>
            <AppRoutes />
        </QueryClientProvider>
    )
}

export default App;