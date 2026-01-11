#include "TCampo.h"

////////////////////////////////////////////////////////////////
// Constructor:                                               //
// * Uff madre mia!, la de cosas que hay que inicializar! xD. //
////////////////////////////////////////////////////////////////
TCampo::TCampo ()
{
   // A inicializar se ha dicho! oooOOoole!  xDD...
   // Primero el tama�o por defecto del campo...
   minX = 0; minY = 3;
   maxX = STDP::TamX() - 1; maxY = STDP::TamY() - 1;

   // ...las caracteristicas del terreno...
   ch = ' '; Atr = STDP_A_NORMAL; CT = STDP_C_NEGRO; CF = STDP_C_VERDE;

   // ...los objetos de la pista...
   Pelota  = new TPelota();
   JugIzq  = new TPaleta(); 
   JugDcha = new TPaleta();

   if (Pelota == NULL || JugIzq == NULL || JugDcha == NULL)
   { 
      cerr <<  "No se ha podido reservar memoria en TCampo." << endl;
      exit (-1);
   }

   Pelota->PonAtributos ('@', STDP_A_NEGRITA, STDP_C_ROJO, CF);
   Pelota->DefineVentana (minX, minY, maxX, maxY);
   Pelota->Crear();

   JugIzq->PonAtributos ('}', STDP_A_NEGRITA, STDP_C_MAGENTA, CF);
   JugIzq->PonLimites   (minY, maxY);
   JugIzq->PonLargo     (3);
   JugIzq->Posicionar   (minX+1, (minY+maxY+3)/2);

   JugDcha->PonAtributos ('{', STDP_A_NEGRITA, STDP_C_AMARILLO, CF);
   JugDcha->PonLimites   (minY, maxY);
   JugDcha->PonLargo     (3);
   JugDcha->Posicionar   (maxX-1, (minY+maxY+3)/2);
}

////////////////////////////////////////////////////////////////
// Destructor:                                                //
// * Destruyeme otra vez, destruyeme otra vez, ven castigame  //
//   con tus variables mas, que memoria guarde para ti... xD  //
////////////////////////////////////////////////////////////////
TCampo::~TCampo ()
{ 
  delete Pelota; delete JugIzq; delete JugDcha; 
  Pelota=NULL; JugIzq=JugDcha=NULL; 
}


////////////////////////////////////////////////////////////////
// Inicializar:                                               //
// * Prepara el campo para comenzar el partido.               //
////////////////////////////////////////////////////////////////
void
TCampo::Inicializar ()
{
   JugIzq->Posicionar    (minX+1, (minY+maxY-JugIzq->Largo())/2);
   JugIzq->PonVelocidad  (0.0);
   JugDcha->Posicionar   (maxX-1, (minY+maxY-JugDcha->Largo())/2);
   JugDcha->PonVelocidad (0.0);
}

////////////////////////////////////////////////////////////////
// PonLimites:                                                //
// * Aqui le decimos cuanto ocupa nuestro campito.            //
////////////////////////////////////////////////////////////////
void
TCampo::PonLimites (int pminX, int pminY, int pmaxX, int pmaxY)
{
   if (pminX < pmaxX && pminY < pmaxY && pminX >= 0 && pminY >=0 &&
       pmaxX < STDP::TamX() && pmaxY < STDP::TamY())
   {  
      minX = pminX; minY = pminY; maxX = pmaxX; maxY = pmaxY;  
      Pelota->DefineVentana (minX, minY, maxX, maxY);
      Pelota->Crear         ();
      JugIzq->PonLimites    (minY, maxY);
      JugDcha->PonLimites   (minY, maxY);
   }
}

////////////////////////////////////////////////////////////////
// PosPelota:                                                 //
// * Varia la posicion de la pelota sobre el campo.           //
////////////////////////////////////////////////////////////////
void
TCampo::PosPelota (int px, int py) { Pelota->Posicionar (px,py); }

////////////////////////////////////////////////////////////////
// TipoPelota:                                                //
// * Con esto le disimos los colores y sabores de la pelota.  //
////////////////////////////////////////////////////////////////
void
TCampo::TipoPelota(chtype pch, int pAtr, int pCT, int pCF)
{
   Pelota->PonAtributos (pch, pAtr, pCT, pCF);
}

////////////////////////////////////////////////////////////////
// TipoPaletas:                                               //
// * Y con esto, pos los de las paletas (fasil, no? xD).      //
////////////////////////////////////////////////////////////////
void
TCampo::TipoPaletas(chtype pch1, int pAtr1, int pCT1, int pCF1,
                    chtype pch2, int pAtr2, int pCT2, int pCF2)
{
   JugIzq->PonAtributos (pch1, pAtr1, pCT1, pCF1);
   JugDcha->PonAtributos (pch2, pAtr2, pCT2, pCF2);
}

////////////////////////////////////////////////////////////////
// TipoCampo:                                                 //
// * Idem, idem, idem... ya sabes... xD.                      //
////////////////////////////////////////////////////////////////
void
TCampo::TipoCampo (chtype pch, int pAtr, int pCT, int pCF)
{
   ch = pch;
   if (STDP::EsAtributo(pAtr)) Atr = pAtr;
   if      (STDP::EsColor(CT)) CT = pCT;
   if (STDP::EsColor(CF)) 
   {
      CF = pCF;
      Pelota->PonAtributos (Pelota->Caracter(),Pelota->Atributo(),
                            Pelota->ColorTexto(),CF);
      JugIzq->PonAtributos (JugIzq->Caracter(),JugIzq->Atributo(),
                            JugIzq->ColorTexto(),CF);      
      JugDcha->PonAtributos(JugDcha->Caracter(),JugDcha->Atributo(),
                            JugDcha->ColorTexto(),CF);
   }
}

////////////////////////////////////////////////////////////////
// LargoPaletas:                                              //
// * Asin las hacemos mas largas o mas chiquiticas xD.        //
////////////////////////////////////////////////////////////////
void
TCampo::LargoPaletas (int largoizq, int largodcha)
{
   JugIzq->PonLargo  (largoizq);
   JugDcha->PonLargo (largodcha);
}

////////////////////////////////////////////////////////////////
// VelPaletas:                                                //
// * Con esto podemos poner la velocidad de las paletas.      //
////////////////////////////////////////////////////////////////
void
TCampo::VelPaletas (float vp1, float vp2)
{
   JugIzq->PonVelocidad  (vp1);
   JugDcha->PonVelocidad (vp2);
}

////////////////////////////////////////////////////////////////
// Dibujar:                                                   //
// * Pintamos el campo y to lo que tiene insima xD.           //
////////////////////////////////////////////////////////////////
void
TCampo::Dibujar ()
{
   STDP::CambiaColor    (Atr, CT, CF);
   STDP::DibujaCuadrado (minX, minY, maxX, maxY, (chtype)ch);
   Pelota->Dibujar();
   JugIzq->Dibujar();
   JugDcha->Dibujar();
}

////////////////////////////////////////////////////////////////
// Mover:                                                     //
// * Pos eso, nos ponemos tos en marcha :P.                   //
////////////////////////////////////////////////////////////////
int
TCampo::Mover (float AcelIzq, float AcelDcha)
{
   int choque;

   JugIzq->AplicaAceleracion (AcelIzq);
   JugDcha->AplicaAceleracion (AcelDcha);
   JugIzq->Mover();
   JugDcha->Mover();
   choque = Pelota->Mover();
   Pelota->CompruebaChoque (*JugIzq);
   Pelota->CompruebaChoque (*JugDcha);

   return (choque);
}

////////////////////////////////////////////////////////////////
// Sacar:                                                     //
// * Realiza un saque de centro.                              //
////////////////////////////////////////////////////////////////
void
TCampo::Sacar (int Direccion)
{ 
   if (Direccion == PORTERIA_IZQ || Direccion == PORTERIA_DCHA)
      Pelota->Crear (0.5 * Direccion);
   else
      Pelota->Crear ();
}

////////////////////////////////////////////////////////////////
// Atributo:                                                  //
// * Poz Devorvemos er atributo del campo, mismamente.        //
////////////////////////////////////////////////////////////////
int
TCampo::Atributo () const { return Atr; }

////////////////////////////////////////////////////////////////
// ColorTexto:                                                //
// * Y tambien, mismamente, el color de texto.		           //
////////////////////////////////////////////////////////////////
int
TCampo::ColorTexto () const { return CT; }

////////////////////////////////////////////////////////////////
// ColorFondo:                                                //
// * Y, como no, nuestro colori�o de fondo.                   //
////////////////////////////////////////////////////////////////
int
TCampo::ColorFondo () const { return CF; }

////////////////////////////////////////////////////////////////
// CentroX:                                                   //
// * Devuerve la posicion horisontal del centro del campo.    //
////////////////////////////////////////////////////////////////
int
TCampo::CentroX () const { return (maxX + minX)/2; }

////////////////////////////////////////////////////////////////
// CentroY:                                                   //
// * Devuerve la posicion vertical del centro del campo.      //
////////////////////////////////////////////////////////////////
int
TCampo::CentroY () const { return (maxY + minY)/2; }

////////////////////////////////////////////////////////////////
// LimiteMaxY:                                                //
// * Devuerve la posicion vertical maxima del campo.          //
////////////////////////////////////////////////////////////////
int
TCampo::LimiteMaxY () const { return maxY; }

////////////////////////////////////////////////////////////////
// LimiteMinY:                                                //
// * Devuerve la posicion vertical minima del campo.          //
////////////////////////////////////////////////////////////////
int
TCampo::LimiteMinY () const { return minY; }

////////////////////////////////////////////////////////////////
// PaletaIzqY:                                                //
// * Devuerve la posicion vertical de la pala izquierda.      //
////////////////////////////////////////////////////////////////
float
TCampo::PaletaIzqY () const { return JugIzq->PosY(); }

////////////////////////////////////////////////////////////////
// PaletaDchaY:                                               //
// * Pos eso mismo, pero de la pala derecha xD.	              //
////////////////////////////////////////////////////////////////
float
TCampo::PaletaDchaY () const { return JugDcha->PosY(); }

////////////////////////////////////////////////////////////////
// PaletaIzqVel:                                              //
// * Pos nos da la velosidad de la pala Izquierda.            //
////////////////////////////////////////////////////////////////
float
TCampo::PaletaIzqVel () const { return JugIzq->VelY(); }

////////////////////////////////////////////////////////////////
// PaletaDchaVel:                                             //
// * Amiga mia no se que decir ni que hacer para verte feliz..//
////////////////////////////////////////////////////////////////
float
TCampo::PaletaDchaVel () const { return JugDcha->VelY(); }


////////////////////////////////////////////////////////////////
// PaletaIzqLargo:                                            //
// * Devuerve lo larga que es la pala izquierda xD.           //
////////////////////////////////////////////////////////////////
int
TCampo::PaletaIzqLargo () const { return JugIzq->Largo(); }

////////////////////////////////////////////////////////////////
// PaletaDchaLargo:                                           //
// * Todo lo que fui es todo lo que soy, con lo que vine ayer //
// es con lo que me voy hoy hoy, yeahhh xDD.                  //
////////////////////////////////////////////////////////////////
int
TCampo::PaletaDchaLargo () const { return JugDcha->Largo(); }

////////////////////////////////////////////////////////////////
// PelotaX:                                                   //
// * Posision horisontal de la pelota ;-D.                    //
////////////////////////////////////////////////////////////////
float
TCampo::PelotaX () const { return Pelota->PosX(); }

////////////////////////////////////////////////////////////////
// PelotaY:                                                   //
// * Vayamos por esa senda a ver que luz encontramos...       //
////////////////////////////////////////////////////////////////
float
TCampo::PelotaY () const { return Pelota->PosY(); }


