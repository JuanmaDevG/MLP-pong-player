#include "TMensaje.h"

///////////////////////////////////////////////////////////
// Constructor:                                          //
// * Aunque parezca mentira me pongo colorada cuando...  //
///////////////////////////////////////////////////////////
TMensaje::TMensaje()
{
   // El tipo de mensaje por defecto es OK
   tipoMensaje = TMENSAJE_T_OK;
   sombra = false;

   // Definimos la ventana por defecto
   minX = STDP::TamX()/2 - 9; maxX = minX + 18;
   minY = STDP::TamY()/2 - 4; maxY = minY + 7;

   // Definimos ventanas de los textos
   titulo.DefineVentana (minX, minY, maxX, maxY);
   texto.DefineVentana (minX, minY, maxX, maxY);

   // Definimos textos por defecto y atributos de los mismos   
   chM=' '; AtrM=STDP_A_NORMAL; CTM=STDP_C_NEGRO; CFM=STDP_C_BLANCO;
   chF=' '; AtrF=STDP_A_NORMAL; CTF=STDP_C_NEGRO; CFF=STDP_C_CIAN;
   titulo.PonCadena    ("Mensaje:");
   titulo.Posicionar   (1, 0);
   titulo.PonAtributos (AtrM, CTM, CFM);
   texto.PonCadena     ("Texto del mensaje");
   texto.Posicionar    (2, 2);
   texto.PonAtributos  (AtrF, CTF, CFF);
}

///////////////////////////////////////////////////////////
// Destructor:                                           //
// * Este mensaje se autodestruira en 3 segundos...      //
///////////////////////////////////////////////////////////
TMensaje::~TMensaje () { /* 3, 2, 1... BOOOooOOOOooOOOooOM! */ }

///////////////////////////////////////////////////////////
// PonTitulo:                                            //
// * Ponemos el titulo de la ventana de mensaje...       //
///////////////////////////////////////////////////////////
void
TMensaje::PonTitulo (const char *tit) 
{ 
   int lon;

   titulo.PonCadena (tit); 
   lon = strlen (tit) + 2;
   if (lon > 80) lon = 80;
   if (lon > maxX - minX + 1) PonAncho (lon);
}

///////////////////////////////////////////////////////////
// PonTexto:                                             //
// * Pos este es er testo que sardra en er menzaje...    //
///////////////////////////////////////////////////////////
void
TMensaje::PonTexto (const char *txt) 
{ 
   int lon;

   texto.PonCadena (txt); 
   lon = strlen (txt) + 4;
   if (lon > 80) lon = 80;
   if (lon > maxX - minX + 1) PonAncho (lon);
}

///////////////////////////////////////////////////////////
// PonAtrTitulo:                                         //
// * Los colorines del titulin...                        //
///////////////////////////////////////////////////////////
void
TMensaje::PonAtrTitulo (int Atr, int CT, int CF) 
{ titulo.PonAtributos (Atr, CT, CF); }

///////////////////////////////////////////////////////////
// PonAtrTexto:                                          //
// * Los colorines del mensajecillo...                   //
///////////////////////////////////////////////////////////
void
TMensaje::PonAtrTexto (int Atr, int CT, int CF) 
{ texto.PonAtributos (Atr, CT, CF); }

///////////////////////////////////////////////////////////
// PonTipo:                                              //
// * Establece el tipo de mensaje (OK, SI_NO, etc...)    //
///////////////////////////////////////////////////////////
void
TMensaje::PonTipo (int tipo) { tipoMensaje = tipo; }

///////////////////////////////////////////////////////////
// PonMarco:                                             //
// * Pone colorines chachiguays al marco.                //
///////////////////////////////////////////////////////////
void
TMensaje::PonMarco (chtype ch, int Atr, int CT, int CF) 
{ 
   chM = ch;
   if (STDP::EsAtributo (Atr)) AtrM = Atr;
   if     (STDP::EsColor (CT)) CTM = CT;
   if     (STDP::EsColor (CF)) CFM = CF;
   titulo.PonAtributos (titulo.Atributo(),titulo.ColorTexto(), CFM);
}

///////////////////////////////////////////////////////////
// PonFondo:                                             //
// * Pone los colorines del fondo.                       //
///////////////////////////////////////////////////////////
void
TMensaje::PonFondo (chtype ch, int Atr, int CT, int CF) 
{ 
   chF = ch;
   if (STDP::EsAtributo (Atr)) AtrF = Atr;
   if     (STDP::EsColor (CT)) CTF = CT;
   if     (STDP::EsColor (CF)) CFF = CF;
   texto.PonAtributos (texto.Atributo(),texto.ColorTexto(), CFF);
}

///////////////////////////////////////////////////////////
// PonAncho:                                             //
// * Pone la anchura del cuadro de mensaje.              //
///////////////////////////////////////////////////////////
void
TMensaje::PonAncho (int anchomens)
{
   if (anchomens > 0 && anchomens <= STDP::TamX())
   {
      minX = (STDP::TamX() - anchomens)/2;
      maxX = minX + anchomens;
      titulo.DefineVentana (minX, minY, maxX, maxY);
      texto.DefineVentana  (minX, minY, maxX, maxY);
      titulo.Posicionar    (1, 0);
      texto.Posicionar     (2, 2);
   }
}

///////////////////////////////////////////////////////////
// PonSombra:                                            //
// * Le ponemos sombra a la ventana para q quede +mona xD//
///////////////////////////////////////////////////////////
void
TMensaje::PonSombra () { sombra = true; }

///////////////////////////////////////////////////////////
// QuitaSombra:                                          //
// * Le quitamos la sombra (oooh! que peeena... xD)      //
///////////////////////////////////////////////////////////
void
TMensaje::QuitaSombra () { sombra = false; }

///////////////////////////////////////////////////////////
// Ejecutar:                                             //
// * Muestra el cuadro de mensaje en pantalla y espera   //
// respuesta de teclado.                                 //
///////////////////////////////////////////////////////////
int
TMensaje::Ejecutar () 
{ 
   // Dibujar
   //--------
   STDP::CambiaColor    (AtrF, CTF, CFF);
   STDP::DibujaCuadrado (minX, minY, maxX, maxY, chF);
   STDP::CambiaColor    (AtrM, CTM, CFM);
   STDP::DibujaMarco    (minX, minY, maxX, maxY, chM);
   if (sombra)
   {
      int n=0;
      STDP::CambiaColor (STDP_A_NORMAL,STDP_C_NEGRO,STDP_C_NEGRO);
      if (maxY + 1 < STDP::TamY())
      { 
         n++;
         STDP::DibujaLineaHorizontal (minX + 1, maxX, maxY + 1, ' ');
      }
      if (maxX + 1 < STDP::TamX())
         STDP::DibujaLineaVertical   (minY + 1, maxY + n, maxX + 1, ' ');
   }
   titulo.Dibujar();
   texto.Dibujar();
   int color = STDP_C_NEGRO;
   if (CFF == STDP_C_NEGRO) color = STDP_C_BLANCO;
   switch (tipoMensaje)
   {
      case TMENSAJE_T_SINO:
      {
         int posx = minX + ((maxX - minX) - 12) / 2;
         int posy = maxY - 2;
         STDP::PonCursor   (posx, posy);
         STDP::CambiaColor (STDP_A_NEGRITA, STDP_C_ROJO, color);
         STDP::EscribeStr  (" S");
         STDP::CambiaColor (STDP_A_NORMAL, STDP_C_AZUL, color);
         STDP::EscribeStr  ("i ");

         STDP::PonCursor   (posx+8, posy);
         STDP::CambiaColor (STDP_A_NEGRITA, STDP_C_ROJO, color);
         STDP::EscribeStr  (" N");
         STDP::CambiaColor (STDP_A_NORMAL, STDP_C_AZUL, color);
         STDP::EscribeStr  ("o ");
         break;
      }
      case TMENSAJE_T_OK:
      default:
      {
         int posx = minX + ((maxX - minX) - 4) / 2;
         int posy = maxY - 2;
         STDP::PonCursor   (posx, posy);
         STDP::CambiaColor (STDP_A_NEGRITA, STDP_C_ROJO, color);
         STDP::EscribeStr  (" O");
         STDP::CambiaColor (STDP_A_NORMAL, STDP_C_AZUL, color);
         STDP::EscribeStr  ("k ");
         break;
      }
   }
   STDP::Refrescar();

   // Comprobar Teclado
   //------------------
   int tecla;
   do
   {
      // Comprobamos teclado
      switch (STDP::ObtenTecla())
      {
         case 'O':
         case 'o':
         {
            tecla = TMENSAJE_R_OK;
            break;
         }
         case 'S':
         case 's':
         {
            tecla = TMENSAJE_R_SI;
            break;
         }
         case 'N':
         case 'n':
         {
            tecla = TMENSAJE_R_NO;
            break;
         }
      }

      // Filtramos segun tipo de mensaje
      switch (tipoMensaje)
      {
         case TMENSAJE_T_SINO:
         {
            if (tecla != TMENSAJE_R_SI && tecla != TMENSAJE_R_NO)
               tecla = ERR;
            break;
         }
         case TMENSAJE_T_OK:
         default:
         {
            if (tecla != TMENSAJE_R_OK) tecla = ERR;
            break;
         }
      }
   } while (tecla == ERR);

   return tecla;
}
