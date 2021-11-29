#include "SystemParameters.h"
#include "pugixml.hpp"
#include <iostream>

SystemParameters::SystemParameters(const std::string inputXmlFile)
{
    pugi::xml_document doc;
    doc.load_file(inputXmlFile.c_str());
    pugi::xml_node systemParameters = doc.child("system");
    params.cart.mass = systemParameters.child("cart").child("mass").text().as_double();
    params.pendulum.mass = systemParameters.child("pendulum").child("mass").text().as_double();
    params.pendulum.length = systemParameters.child("pendulum").child("length").text().as_double();
    params.pendulum.initialAngle = systemParameters.child("pendulum").child("initial_angle").text().as_double();
    params.pendulum.momentOfInertiaAboutCenter = params.pendulum.mass * params.pendulum.length * params.pendulum.length / 12;
    params.gravity = systemParameters.child("gravity").text().as_double();
    params.friction = systemParameters.child("friction").text().as_double();

}
