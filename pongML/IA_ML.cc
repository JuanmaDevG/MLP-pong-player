#include "IA_ML.h"
#include "TPartido.h"
#include <fstream>
#include <limits>
#include <vector>

///////////////////////////////////////////////////////////
// Inteligencia:                                         //
// * Vectores de Pesos de los perceptrones para la IA    //
///////////////////////////////////////////////////////////
static constexpr auto MAX_SIZE = std::numeric_limits<std::streamsize>::max();
const unsigned s_kinputs = s_N_EstadosJuego-1;
struct AI { double w[2][s_kinputs]{}; };
std::vector<AI> AIs{};

///////////////////////////////////////////////////////////
// Funcion Obtener Pesos de IAs                          //
// * Lee los pesos de las IAs de un fichero              //
///////////////////////////////////////////////////////////
void
EliminaComentarios(std::ifstream& in) {
   while(true) {
      auto const c = in.get();
      if (in.eof() || in.bad()) break;
      if (c != '#') break;
      // Comment found: remove
      in.ignore(MAX_SIZE, '\n');
   }
   in.unget();
}

AI
LeeUnaIA(std::ifstream& in) {
   AI one_ai{};

   // Read one AI 
   for(auto j{0uz}; j < 2; j+=1) {
      for(auto i{0uz}; i < s_kinputs; i+=1) {
         in >> one_ai.w[j][i];
      }
      in.ignore(MAX_SIZE, '\n');
   }

   return one_ai;
}

void
ImprimeUnaIA(std::ostream& out, AI const& ai) {
   for(auto j{0uz}; j < 2; j+=1) {
      for(auto i{0uz}; i < s_kinputs; i+=1) {
         out << ai.w[j][i] << " || ";
      }
      out << '\n';
   }
}

void 
TPartido::LeerPesosIAs(std::string const& fichero) {
   std::ifstream in(fichero);

   while(in.is_open() && in.good()) {
      EliminaComentarios(in);
      auto ai = LeeUnaIA(in);
      AIs.push_back(ai);

      //DEBUG (Comprueba que se leen las IAs correctamente)
      //std::cerr << "IA LEIDA: " << AIs.size()-1 << "\n";
      //ImprimeUnaIA(std::cerr, ai);
   }
}

///////////////////////////////////////////////////////////
// Funcion Hipótesis (H):                                //
// * Calcula la hipótesis (1/0) según estado y pesos     //
// * Asume inputs(x) sin coordenada x_0                  //
// * Asume pesos(w) con bias (w_0) el primero            //
///////////////////////////////////////////////////////////
int sign(auto const x) {
   return (x >= 0) ? 1 : 0;
}
double h(double const* w, double const* x) {
   // Iniciar con bias
   double hval = w[0];

   // Sumar wx de 1 a n
   for(unsigned i=1; i < s_kinputs; i+=1)
      hval += x[i-1] * w[i];

   // Añadir w0x0 (1*x0), calcular el signo y devolver
   return hval;
}


///////////////////////////////////////////////////////////
// INTELIGENCIA DE LAS PALAS                             //
// * Versión controlada por Machine Learning             //
///////////////////////////////////////////////////////////
#ifdef MACHINE_LEARNING
void
TPartido::IniciarInteligencias (std::string const& fichero) {
   LeerPesosIAs(fichero);
}

double
TPartido::Inteligencia (int jugador) {
   double aum {0.0};
   unsigned bup, bdown;

   if (jugador == 1) {
      bup   = sign( h( AIs[IA_Izq].w[0], vEstadoJuego ) );
      bdown = sign( h( AIs[IA_Izq].w[1], vEstadoJuego ) );
   } else {
      double vEstadoJ_2[s_N_EstadosJuego];

      vEstadoJ_2[s_jugadorVY]   = campo.PaletaDchaVel();
      vEstadoJ_2[s_enemigoYRel] = campo.PaletaIzqY() - campo.PaletaDchaY();
      vEstadoJ_2[s_enemigoVY]   = campo.PaletaIzqVel();
      vEstadoJ_2[s_pelotaXRel]  = campo.PaletaDchaX() - campo.PelotaX();
      vEstadoJ_2[s_pelotaYRel]  = campo.PelotaY() - campo.PaletaDchaY();
      vEstadoJ_2[s_pelotaVX]    = -campo.PelotaVX();
      vEstadoJ_2[s_pelotaVY]    = campo.PelotaVY();

      bup   = sign( h( AIs[IA_Der].w[0], vEstadoJ_2 ) );
      bdown = sign( h( AIs[IA_Der].w[1], vEstadoJ_2 ) );
   }

   if      ( bup && !bdown) aum = -0.2;      
   else if (!bup &&  bdown) aum = +0.2;

   return aum;
}
#endif

///////////////////////////////////////////////////////////
// maxIA:                                                //
// * Devuelve el máximo número válido de inteligencia    //
// Machine Learning Disponible                           //
///////////////////////////////////////////////////////////
unsigned
TPartido::maxIA() const {
   return AIs.size() - 1;
}
